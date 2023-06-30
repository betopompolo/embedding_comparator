from dataclasses import dataclass
from typing import List
from bson import ObjectId
import more_itertools
from tqdm import tqdm
from code_search_net_repository import CodeSearchNetProgrammingLanguage, CodeSearchNetPartition, CodeSearchNetRepository
from models import MongoDbPairDoc, Runnable

from mongo_db_client import MongoDbClient


@dataclass
class ImportCSNetPairsToMongoDb(Runnable):
  def run(self):
    mongo_db_client = MongoDbClient()
    batch_size = 128
    pairs_collection = mongo_db_client.get_pairs_collection()
    repo = CodeSearchNetRepository()
    partitions: List[CodeSearchNetPartition] = ['test', 'train', 'valid']
    languages: List[CodeSearchNetProgrammingLanguage] = ['java', 'python']

    
    for language in languages:
      for partition in partitions:
        with tqdm(desc=f"Importing {language} pairs from {partition} partition to MongoDB", total=repo.get_count_for(language, partition)) as progress_bar:
          for cs_net_pairs in more_itertools.chunked(repo.read_dataset(language, partition), batch_size) :
            documents: List[MongoDbPairDoc] = [{
              '_id': ObjectId(),
              'code_tokens': pair['code_tokens'],
              'comment_tokens': pair['comment_tokens'],
              'github_url': pair['url'],
              'language': pair['language'],
              'partition': pair['partition']
            } for pair in cs_net_pairs]
            pairs_collection.insert_many(documents)

            progress_bar.update(len(documents))

# TODO: Runnable for queries (raw and 'cs_query')
def write_queries():
  pass
  # collection = db_client.get_queries_collection()
  # query_dataset = CSNetQueryDataset()

  # with tqdm(desc="Write queries.csv into mongo database", total=query_dataset.get_dataset_count()) as progress_bar:
  #   for queries_batch in more_itertools.chunked(query_dataset.get_dataset(), batch_size):
  #     documents = [{
  #       "query": query.query,
  #       "language": query.language,
  #       "relevance": query.relevance,
  #       "url": query.url,
  #     } for query in queries_batch]

  #     collection.insert_many(documents)
  #     progress_bar.update(len(documents))
