import more_itertools
from pymongo import MongoClient
from pymongo.database import Database as MongoDatabase
from tqdm import tqdm

from code_search_net_dataset import CodeSearchNetDataset
from orjson_parser import OrJsonParser
from models import CodeCommentPair
from query_dataset import QueryDataset

mongo_client = MongoClient('mongodb://127.0.0.1:27018/')
cs_net_database = mongo_client['cs_net_mongo']
batch_size = 128

def get_collection(collection_name: str):
  return cs_net_database.get_collection(collection_name) if collection_name in cs_net_database.list_collection_names() else cs_net_database.create_collection(collection_name)

def write_pairs():
  pairs_collection = get_collection('pairs')
  cs_net = CodeSearchNetDataset(
    json_parser=OrJsonParser()
  )

  with tqdm(desc="Write code/comment pairs into mongo database", total=cs_net.get_dataset_count()) as progress_bar:
    for pairs_batch in more_itertools.chunked(cs_net.get_dataset(), batch_size):
      documents = [{
        "github_url": pair.id,
        "code_tokens": pair.code_tokens,
        "comment_tokens": pair.comment_tokens,
        "partition": pair.partition
      } for pair in pairs_batch]
      pairs_collection.insert_many(documents)

      progress_bar.update(len(documents))

def write_queries():
  collection = get_collection('queries')
  query_dataset = QueryDataset()

  with tqdm(desc="Write queries.csv into mongo database", total=query_dataset.get_dataset_count()) as progress_bar:
    for queries_batch in more_itertools.chunked(query_dataset.get_dataset(), batch_size):
      documents = [{
        "query": query.query,
        "language": query.language,
        "relevance": query.relevance,
        "url": query.url,
      } for query in queries_batch]

      collection.insert_many(documents)
      progress_bar.update(len(documents))
