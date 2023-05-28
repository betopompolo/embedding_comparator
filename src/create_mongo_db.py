# TODO: Make it simpler and without CodeCommentPair abstraction
import more_itertools
from tqdm import tqdm

from cs_net_datasets import CSNetPairDataset, CSNetQueryDataset
from models import CodeCommentPair, DatasetRepository
from mongo_db_client import MongoDbClient
from orjson_parser import OrJsonParser

db_client = MongoDbClient()
batch_size = 128

def write_pairs():
  pairs_collection = db_client.get_pairs_collection()
  pairs_repo: DatasetRepository[CodeCommentPair] = CSNetPairDataset(
    json_parser=OrJsonParser(),
  )

  with tqdm(desc="Write code/comment pairs into mongo database", total=pairs_repo.get_dataset_count()) as progress_bar:
    for pairs_batch in more_itertools.chunked(pairs_repo.get_dataset(), batch_size):
      documents = [{
        "github_url": pair.id,
        "code_tokens": pair.code_tokens,
        "comment_tokens": pair.comment_tokens,
        "partition": pair.partition,
        "language": pair.language,
      } for pair in pairs_batch]
      pairs_collection.insert_many(documents)

      progress_bar.update(len(documents))

def write_queries():
  collection = db_client.get_queries_collection()
  query_dataset = CSNetQueryDataset()

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


# write_pairs()
# write_queries()
