from dataclasses import dataclass
from typing import List
from bson import ObjectId
import more_itertools
from tqdm import tqdm
from code_search_net_pairs_repository import CodeSearchNetPairsRepository
from models import Language, MongoDbPairDoc, Partition, Runnable

from mongo_db_client import MongoDbClient


@dataclass
class ImportCSNetPairsToMongoDb(Runnable):
  def run(self):
    mongo_db_client = MongoDbClient()
    batch_size = 128
    pairs_collection = mongo_db_client.get_pairs_collection()
    repo = CodeSearchNetPairsRepository()
    partitions: List[Partition] = ['test', 'train', 'valid']
    languages: List[Language] = ['python']

    
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
