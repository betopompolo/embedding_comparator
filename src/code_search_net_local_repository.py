import os
from dataclasses import dataclass
from typing import Iterable, List
from pymongo import MongoClient

from models import CodeCommentPair, DatasetRepository, Partition


@dataclass
class CodeSearchNetLocalRepository(DatasetRepository[CodeCommentPair]):
  mongo_client = MongoClient('mongodb://127.0.0.1:27018/')
  cs_net_database = mongo_client['cs_net_mongo']
  
  def get_dataset(self) -> Iterable[CodeCommentPair]:
    pairs_collection = self.cs_net_database.get_collection('pairs')
    for doc in pairs_collection.find():
      yield self.__map_from_db(doc)

  def search(self, github_url: str) -> CodeCommentPair | None:
    pairs_collection = self.cs_net_database.get_collection('pairs')
    db_pair = pairs_collection.find_one(filter={ "github_url": github_url })

    return None if db_pair is None else self.__map_from_db(db_pair)

  def get_dataset_count(self) -> int:
    pairs_collection = self.cs_net_database.get_collection('pairs')
    return pairs_collection.count_documents(filter={})

  def __map_from_db(self, db_pair: dict):
    return CodeCommentPair(
      id=db_pair['github_url'],
      code_tokens=db_pair['code_tokens'],
      comment_tokens=db_pair['comment_tokens'],
      partition=db_pair['partition'],
      language=db_pair['language'],
    )
