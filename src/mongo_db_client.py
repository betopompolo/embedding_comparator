from typing import List, TypedDict
from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection

from cs_net_parser import CSNetCodeLanguage, CSNetPartition

MongoId = ObjectId
class MongoDbPairDoc(TypedDict):
  _id: MongoId
  id: str
  code_tokens: List[str]
  comment_tokens: List[str]
  github_url: str
  language: CSNetCodeLanguage
  partition: CSNetPartition

class MongoDbQueryDoc(TypedDict):
  pair_id: MongoId
  language: CSNetCodeLanguage
  github_url: str
  query: str
  relevance: int
  notes: str

class MongoDbClient:
  mongo_client = MongoClient('mongodb://127.0.0.1:27018/')
  cs_net_database = mongo_client['code_search_net']
  
  def get_pairs_collection(self) -> Collection[MongoDbPairDoc]:
    return self.cs_net_database.get_collection('pairs')
  
  def get_queries_collection(self) -> Collection[MongoDbQueryDoc]:
    return self.cs_net_database.get_collection('queries')

