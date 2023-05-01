import os
from dataclasses import dataclass
from typing import Iterable, Optional
from pymongo import MongoClient

import tensorflow as tf

from models import CodeCommentPair, DatasetRepository, JsonParser
from utils import codesearchnet_dataset_len, decode_tensor_string


@dataclass
class CodeSearchNetDataset(DatasetRepository[CodeCommentPair]):
  mongo_client = MongoClient('mongodb://127.0.0.1:27018/')
  cs_net_database = mongo_client['cs_net_mongo']
  
  def get_dataset(self) -> Iterable[CodeCommentPair]:
    return super().get_dataset()

  def search(self, github_url: str) -> CodeCommentPair | None:
    pairs_collection = self.cs_net_database.get_collection('pairs')
    db_pair = pairs_collection.find_one(filter={ "github_url": github_url })

    return None if db_pair is None else self.__map_from_db(db_pair)

  def get_dataset_count(self) -> int:
    count = 0
    for key in codesearchnet_dataset_len:
      count += sum(codesearchnet_dataset_len[key].values())
    return count

  def __map_from_db(self, db_pair: dict):
    return CodeCommentPair(
      id=db_pair['github_url'],
      code_tokens=db_pair['code_tokens'],
      comment_tokens=db_pair['comment_tokens'],
      partition=db_pair['partition'],
    )
