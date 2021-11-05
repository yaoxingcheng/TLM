import logging
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.modeling_outputs import MaskedLMOutput
from transformers.utils.dummy_pt_objects import NoRepeatNGramLogitsProcessor

logger = logging.getLogger(__name__)

class BertForMaskedLM(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, set_cls=False, num_labels=0):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.cls_layer = None
        if set_cls:
            self.cls_layer = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, num_labels),
            )

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    
    def set_cls_layer(self, num_labels, config):
        self.cls_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, num_labels),
        )
    def set_args(self, args):
        self.args = args

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_loss=True,
        cls_labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        cls_prediction_scores = self.cls_layer(sequence_output[:,0])

        # compute cls loss
        cls_loss = None
        cls_weight = 0.0
        if cls_labels is not None:
            cls_loss = loss_fct(cls_prediction_scores, cls_labels.view(-1))
            cls_weight = 1.0
        else:
            cls_loss = cls_prediction_scores.sum() * 0.0

        # compute mlm loss
        mlm_loss = None
        if labels is not None and (self.args.mlm_weight > 0.0 or cls_labels is None):
            assert labels is not None, "mlm without labels!"
            prediction_scores = self.cls(sequence_output)
            l_shape = labels.shape
            mlm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        else:
            mlm_loss = self.cls(sequence_output).sum() * 0.0
        
        if self.args.mlm_weight == 0 and cls_labels is None:
            mlm_weight = 1
        else:
            mlm_weight = self.args.mlm_weight

        loss = (mlm_loss * self.args.mlm_weight + cls_loss * cls_weight) / (mlm_weight + cls_weight)
        
        return MaskedLMOutput(
            loss=loss,
            logits=cls_prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )