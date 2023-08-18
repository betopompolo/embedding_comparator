from dataclasses import dataclass
from typing import List, cast
from bson import ObjectId
import more_itertools
from tqdm import tqdm
from code_search_net_queries_repository import CodeSearchNetQueriesRepository, CodeSearchNetQueryProgrammingLanguage
from models import Language, QueryDbDoc, Runnable

from mongo_db_client import MongoDbClient


@dataclass
class ImportCSNetQueriesToMongoDb(Runnable):
  def run(self):
    mongo_db_client = MongoDbClient()
    if self.is_pairs_collection_created(mongo_db_client) == False:
      raise RuntimeError('pairs collection was not found in MongoDb')
    
    batch_size = 128
    queries_collection = mongo_db_client.get_queries_collection()
    cs_net_queries_repo = CodeSearchNetQueriesRepository()
    languages: List[CodeSearchNetQueryProgrammingLanguage] = ['Python']
    
    for language in languages:
      with tqdm(desc=f"Importing code search net {language} queries to MongoDB", total=cs_net_queries_repo.get_count(language)) as progress_bar:
        for cs_net_queries in more_itertools.chunked(cs_net_queries_repo.read_queries(language), batch_size):
          documents: List[QueryDbDoc] = []

          for query in cs_net_queries:
            pair_doc = mongo_db_client.get_pairs_collection().find_one({
              "github_url": query['GitHubUrl']
            })

            if pair_doc is None:
              progress_bar.update(1)
              continue

            documents.append({
              '_id': ObjectId(),
              'language': cast(Language, query['Language'].lower()),
              'query': query['Query'],
              'relevance': query['Relevance'],
              'url': query['GitHubUrl'],
              'pair_doc': pair_doc,
            })
          
          queries_collection.insert_many(documents)
          progress_bar.update(len(documents))

  def is_pairs_collection_created(self, db_client: MongoDbClient):
    return db_client.pairs_collection_name in db_client.cs_net_database.list_collection_names()