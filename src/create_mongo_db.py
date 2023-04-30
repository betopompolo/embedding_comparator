import more_itertools
from pymongo import MongoClient
from tqdm import tqdm

from code_search_net_dataset import CodeSearchNetDataset
from json_parser import OrJsonParser
from models import CodeCommentPair

mongo_client = MongoClient('mongodb://127.0.0.1:27018/')
cs_net_database = mongo_client['cs_net_mongo']
pairs_collection = cs_net_database.create_collection('pairs')
cs_net = CodeSearchNetDataset(
  json_parser=OrJsonParser()
)

def pair_to_document(pair: CodeCommentPair):
  return {
    "github_url": pair.id,
    "code_tokens": pair.code_tokens,
    "comment_tokens": pair.comment_tokens,
    "partition": pair.partition
  }

with tqdm(desc="Write code/comment pairs in mongo database", total=cs_net.get_dataset_count()) as progress_bar:
  for pairs_batch in more_itertools.chunked(cs_net.get_dataset(), 128):
    batch_count = len(pairs_batch)
    pairs_collection.insert_many([pair_to_document(pair) for pair in pairs_batch])
    progress_bar.update(batch_count)
