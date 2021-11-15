import sys, os
import torch
import time
import argparse
from tqdm import tqdm
import numpy as np
import datasets
from datasets import load_dataset, load_metric
import pandas as pd
from rake_nltk import Rake

from elasticsearch import Elasticsearch

es = Elasticsearch(
    [{"host": "localhost", "port": "9200"}],
    timeout=300,
)
es_config = {
    "settings": {
        "number_of_shards": 1,
        "analysis": {"analyzer": {"stop_standard": {"type": "standard", " stopwords": "_english_"}}},
    },
    "mappings": {
        "properties": {
            "text": {
                "type": "text",
                "analyzer": "standard",
                "similarity": "BM25"
            },
        }
    },
}

from multiprocessing import Pool

glob_data_source=None
glob_top_k=None
glob_logger=None
glob_text_column_name=None

def get_neighbour_examples(query):
    try:
        _, hard_example = glob_data_source.get_nearest_examples(
            glob_text_column_name,
            query,
            k=glob_top_k
        )
    except:
        if glob_logger is not None:
            glob_logger.info("error detected when search for query {}, skipped this query".format(query))
        hard_example = {"id":[]}
    return hard_example

class BM25Selector:

    def __init__(self, source, target, index_name, num_proc=4, enable_rake=False):
        self.source = source
        self.target = target
        self.num_proc = num_proc
        global glob_data_source
        glob_data_source = source
        global glob_text_column_name
        glob_text_column_name = "text"
        self.text_column_name = "text"
        self.index_name = index_name
        self.enable_rake = enable_rake
        if not es.indices.exists(index=self.index_name):
            print("index name not exist, creating new index for the training corpora")
            self.source.add_elasticsearch_index(
                self.text_column_name,
                es_client=es,
                es_index_name=self.index_name,
                es_index_config=es_config,
            )
            glob_data_source = self.source
        else:
            self.source.load_elasticsearch_index(
                self.text_column_name,
                es_client=es,
                es_index_name=self.index_name
            )
            glob_data_source = self.source
    
    def build_queries(self):
        queries = []
        if self.enable_rake:
            r = Rake()
            for idx in tqdm(range(len(self.target))):
                r.extract_keywords_from_text(self.target[idx]["text"])
                phrases = r.get_ranked_phrases_with_scores()
                # select top 20 key phrases
                query = " ".join([p[1] for p in phrases[:20]])
                queries.append(query)
        else:
            queries = [self.target[idx]["text"] for idx in tqdm(range(len(self.target)))]
        return queries

    def build_dataset(self, top_k, output_file_path, batch_size=4):
        global glob_top_k
        glob_top_k = top_k
        queries = self.build_queries()
        query_neighbours = []
        query_ranks = []
        for i in tqdm(range(0, len(queries), batch_size)):
            if i + batch_size >= len(queries):
                batched_queries = queries[i:]
            else:
                batched_queries = queries[i:i+batch_size]
            with Pool(processes=self.num_proc) as pool:
                results = pool.map(get_neighbour_examples, batched_queries)
                for result in results:
                    query_neighbours.extend(result["id"])
                    query_ranks.extend(list(range(len(result["id"]))))
        
        unique_query = {}
        for nid, rank in zip(query_neighbours, query_ranks):
            if nid not in unique_query:
                unique_query[nid] = rank
            else:
                unique_query[nid] = min(unique_query[nid], rank)
        
        query_neighbours = []
        ranks = []
        for nid, rank in unique_query.items():
            query_neighbours.append(nid)
            ranks.append(rank)
        texts = [self.source[idx][self.text_column_name] for idx in query_neighbours]
        ids = list(range(len(texts)))
        df = pd.DataFrame({"text":texts, "id":ids, "rank":ranks})
        df.to_csv(output_file_path, index=False)

def get_text_dataset(dataset_name):
    data_files = {}
    data_files["train"] = dataset_name
    extension = dataset_name.split(".")[-1]
    if extension == "txt":
            extension = "text"
    raw_datasets = load_dataset(extension, data_files=data_files)
    return raw_datasets["train"]

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--index_name', default="example_bm25_source", type=str)
    parser.add_argument('--source_file', default="./example_data/source.csv", type=str)
    parser.add_argument('--target_file', default="./example_data/target.csv", type=str)
    parser.add_argument('--output_dir', default="./example_data/", type=str)
    parser.add_argument('--output_name', default="selected.csv", type=str)
    parser.add_argument('--top_k', default=50, type=int)
    parser.add_argument('--rake', action="store_true", help="extract key phrases in the query to speed up searching process")
    args = parser.parse_args()

    source_dataset = get_text_dataset(args.source_file)
    target_dataset = get_text_dataset(args.target_file)
    selector = BM25Selector(source_dataset, target_dataset, args.index_name, enable_rake=args.rake)

    output_file_path = os.path.join(args.output_dir, args.output_name)
    selector.build_dataset(args.top_k, output_file_path)
