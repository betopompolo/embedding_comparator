from pymongo import MongoClient
from pymongo.collection import Collection

from models import PairDbDoc, QueryDbDoc, QueryDbDoc

class MongoDbClient:
  mongo_client = MongoClient('mongodb://127.0.0.1:27018/')
  cs_net_database = mongo_client['cs_net_mongo']
  
  def get_pairs_collection(self) -> Collection[PairDbDoc]:
    return self.cs_net_database.get_collection('pairs')
  
  def get_queries_collection(self) -> Collection[QueryDbDoc]:
    return self.cs_net_database.get_collection('queries')

