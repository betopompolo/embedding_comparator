from pymongo import MongoClient
from pymongo.collection import Collection

from models import MongoDbPairDoc, QueryDbDoc, QueryDbDoc

class MongoDbClient:
  mongo_client = MongoClient('mongodb://127.0.0.1:27018/')
  cs_net_database = mongo_client['code_search_net']
  pairs_collection_name = 'pairs'
  
  def get_pairs_collection(self) -> Collection[MongoDbPairDoc]:
    return self.cs_net_database.get_collection(self.pairs_collection_name)
  
  def get_queries_collection(self) -> Collection[QueryDbDoc]:
    return self.cs_net_database.get_collection('queries')

