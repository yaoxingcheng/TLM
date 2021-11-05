import os
import shutil
import json
import torch
import math
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class Trainer:

    def __init__(self,
                 args,
                 model,
                 optimizer,
                 lr_scheduler,
                 train_dataloader,
                 eval_dataloader,
                 external_dataloader,
                 logger,
                 accelerator,
                 metric,
                 label_list,
                 tokenizer,
                 from_checkpoint=None,
                 test_dataloader=None,
                ):
        
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.external_dataloader = external_dataloader
        self.logger = logger
        self.completed_steps = 0
        self.best_metric = 0.0
        self.test_results = 0.0
        self.accelerator = accelerator
        self.label_list = label_list
        self.metric = metric
        self.tokenizer = tokenizer
        self.writter = None
        if self.accelerator.is_main_process and self.args.output_dir is not None:
            summary_path = os.path.join(self.args.output_dir, "summary")
            if os.path.exists(summary_path):
                shutil.rmtree(summary_path)
            self.writter = SummaryWriter(summary_path)
        self._train_iter = iter(train_dataloader)
        if external_dataloader is not None:
            self._external_iter = iter(external_dataloader)
        else:
            self._external_iter = None
        self.from_checkpoint = from_checkpoint
    
    def _move_to_device(self, batch):
        for k in batch:
            batch[k] = batch[k].to(self.args.device)
        return batch
    
    def _save_model(self, save_path=None):
        if save_path is None:
            save_path = self.args.output_dir
        
        if self.accelerator.is_main_process:
           unwrapped_model = self.accelerator.unwrap_model(self.model)
           unwrapped_model.save_pretrained(save_path, save_function=self.accelerator.save)

    def _save_trained(self):
        self._save_model()
        torch.save(self.optimizer.state_dict(), os.path.join(self.args.output_dir, "optimizer.pt"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(self.args.output_dir, "scheduler.pt"))
        trainer_state = {
            "completed_steps": self.completed_steps,
            "best_metric": self.best_metric,
            "test_results": self.test_results,
        }
        if self.accelerator.is_main_process:
            with open(os.path.join(self.args.output_dir, "trainer_state.json"), "w") as f:
                json.dump(trainer_state, f)

    def evaluate(self):
        self.model.eval()
        for eval_step, batch in enumerate(self.eval_dataloader):
            batch = self._move_to_device(batch)
            outputs = self.model(**batch, return_dict=True, return_loss=False)
            label_id_list = [label for label in self.label_list]
            logits = outputs.logits
            predictions = logits.argmax(dim=-1)
            ref = batch["cls_labels"]
            self.metric.add_batch(
                predictions=predictions,
                references=ref,
            )
        eval_metric = self.metric.compute()
        self.logger.info(f"step {self.completed_steps}: {eval_metric}")
        mean_metric = 0.0
        if self.accelerator.is_main_process:
            for key in eval_metric:
                if self.writter is not None:
                    self.writter.add_scalar(f"eval/dev-{key}", eval_metric[key], self.completed_steps)
                mean_metric += eval_metric[key]
        mean_metric /= len(eval_metric)
        self.accelerator.wait_for_everyone()
        is_best = False
        if self.best_metric < mean_metric:
            self.best_metric = mean_metric
            is_best = True
        if self.test_dataloader is not None:
            for eval_step, batch in enumerate(self.test_dataloader):
                batch = self._move_to_device(batch)
                outputs = self.model(**batch, return_dict=True, return_loss=False)
                label_id_list = [label for label in self.label_list]
                logits = outputs.logits
                predictions = logits.argmax(dim=-1)
                ref = batch["cls_labels"]
                self.metric.add_batch(
                    predictions=predictions,
                    references=ref,
                )
            eval_metric = self.metric.compute()
            self.logger.info(f"step {self.completed_steps} test: {eval_metric}")
            if self.accelerator.is_main_process:
                for key in eval_metric:
                    if self.writter is not None:
                        self.writter.add_scalar(f"eval/test-{key}", eval_metric[key], self.completed_steps)
                    self.test_results = eval_metric[key]
        if is_best and self.args.output_dir is not None:
            self.logger.info("best metric on dev set achieved, try to save model checkpoint to {}".format(self.args.output_dir))
            self._save_trained()
        self.model.train()
    
    def _get_next_batch(self):
        try:
            batch = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_dataloader)
            batch = next(self._train_iter)
        return batch
    
    def _get_next_external_batch(self):
        try:
            batch = next(self._external_iter)
        except StopIteration:
            self._external_iter = iter(self.external_dataloader)
            batch = next(self._external_iter)
        return batch
    
    def _get_batch(self, ratio):
        if ratio <= 1 or (self.completed_steps + 1) % ratio == 0:
            batch = self._get_next_batch()
        else:
            batch = self._get_next_external_batch()
        return self._move_to_device(batch)
    
    def compute_loss(self, ratio):
        self.model.train()
        batch = self._get_batch(ratio)
        outputs = self.model(**batch)
        loss = outputs.loss
        loss = loss / self.args.gradient_accumulation_steps
        self.accelerator.backward(loss)
        return loss.item()

    def _prepare_from_checkpoint(self):
        
        if self.from_checkpoint is None:
            return
        
        state_file = os.path.join(self.from_checkpoint, "trainer_state.json")
        optim_file = os.path.join(self.from_checkpoint, "optimizer.pt")
        sched_file = os.path.join(self.from_checkpoint, "scheduler.pt")
        if os.path.exists(sched_file):
            sched_state = torch.load(sched_file)
            self.lr_scheduler.load_state_dict(sched_state)
        if not os.path.exists(state_file):
            return

        with open(state_file, "r") as f:
            state = json.load(f)
            self.pre_completed_steps = state["completed_steps"]
            self.best_metric = state["best_metric"]
        
        self.logger.info(f"pretrained steps: {self.pre_completed_steps}, best dev metric {self.best_metric}")
        self.accelerator.wait_for_everyone()
    
    def update(self, tr_loss, loss_step):
        if self.completed_steps % self.args.steps_to_log == 0:
            self.logger.info(
                "step {}, learning rate {}, average loss {}".format(
                    self.completed_steps,
                    self.optimizer.param_groups[0]["lr"],
                    tr_loss / loss_step
                )
            )
            if self.accelerator.is_main_process:
                if self.writter is not None:
                    self.writter.add_scalar('train/loss', tr_loss / loss_step, self.completed_steps)
                    self.writter.add_scalar('train/lr', self.optimizer.param_groups[0]["lr"], self.completed_steps)
            tr_loss = 0.0
            loss_step = 0
        
        if self.completed_steps % self.args.steps_to_eval == 0:
            self.evaluate()
        
        if self.completed_steps % self.args.steps_to_save == 0:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self._save_model(
                    save_path = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(self.completed_steps))
                )
                # delete outdated checkpoints
                for files in os.listdir(self.args.output_dir):
                    file_name = os.path.join(self.args.output_dir, files)
                    if os.path.isdir(file_name) and files.startswith('checkpoint-'):
                        checked_step = int(files[11:])
                        if self.completed_steps - checked_step >= self.args.max_ckpts_to_keep * self.args.steps_to_save:
                            if self.accelerator.is_main_process:
                                shutil.rmtree(file_name)

    def train(self):

        total_batch_size = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.accelerator.num_processes
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.args.max_train_steps // self.args.gradient_accumulation_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.args.max_train_steps // self.args.gradient_accumulation_steps))
        self.completed_steps = 0
        self.pre_completed_steps = 0
        self._prepare_from_checkpoint()
        for step in range(self.args.max_train_steps):
            if self.completed_steps < self.pre_completed_steps:
                self._get_batch(self.args.external_ratio)
                self.completed_steps += 1
                progress_bar.update(1)
                continue
            tr_loss += self.compute_loss(self.args.external_ratio)
            loss_step += 1
            if step % self.args.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
                self.completed_steps += 1
                self.update(tr_loss, loss_step)
                tr_loss = 0.0
                loss_step = 0
        
        self.evaluate()
        if self.args.save_final and self.args.output_dir is not None:
            self._save_model(
                save_path = os.path.join(self.args.output_dir, 'final')
            )