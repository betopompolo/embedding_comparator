from dataclasses import dataclass
from typing import Iterable, List

from models import CodeCommentPair, DatasetRepository, Query
from mongo_db_client import MongoDbClient


@dataclass
class TrainingPairsRepository(DatasetRepository[CodeCommentPair]):
  db_client = MongoDbClient()
  take: int
  
  def get_dataset(self) -> Iterable[CodeCommentPair]:
    pairs_collection = self.db_client.get_collection('pairs')
    take = int(self.take / 2)

    python_pairs = [self.__map_from_db(item) for item in pairs_collection.find({ "language": "python", "partition": 'train' }).limit(take)] 
    java_pairs = [self.__map_from_db(item) for item in pairs_collection.find({ "language": "java", "partition": 'train' }).limit(take)]

    return python_pairs + java_pairs

  def search(self, github_url: str) -> CodeCommentPair | None:
    pairs_collection = self.db_client.get_collection('pairs')
    db_pair = pairs_collection.find_one(filter={ "github_url": github_url })

    return None if db_pair is None else self.__map_from_db(db_pair)

  def get_dataset_count(self) -> int:
    pairs_collection = self.db_client.get_collection('pairs')
    return pairs_collection.count_documents(filter={})

  def __map_from_db(self, db_pair: dict):
    return CodeCommentPair(
      id=db_pair['github_url'],
      code_tokens=db_pair['code_tokens'],
      comment_tokens=db_pair['comment_tokens'],
      partition=db_pair['partition'],
      language=db_pair['language'],
    )
  
@dataclass
class ValidationPairsRepository(DatasetRepository[CodeCommentPair]):
  db_client = MongoDbClient()
  
  def get_dataset(self) -> Iterable[CodeCommentPair]:
    pairs_collection = self.db_client.get_collection('pairs')
    for doc in pairs_collection.find():
      yield self.__map_from_db(doc)

  def search(self, github_url: str) -> CodeCommentPair | None:
    pairs_collection = self.db_client.get_collection('pairs')
    db_pair = pairs_collection.find_one(filter={ "github_url": github_url })

    return None if db_pair is None else self.__map_from_db(db_pair)

  def get_dataset_count(self) -> int:
    pairs_collection = self.db_client.get_collection('pairs')
    return pairs_collection.count_documents(filter={})

  def __map_from_db(self, db_pair: dict):
    return CodeCommentPair(
      id=db_pair['github_url'],
      code_tokens=db_pair['code_tokens'],
      comment_tokens=db_pair['comment_tokens'],
      partition=db_pair['partition'],
      language=db_pair['language'],
    )

  
@dataclass
class QueriesRepository(DatasetRepository[Query]):
  db_client = MongoDbClient()
  take: int | None = None
  
  def get_dataset(self) -> Iterable[Query]:
    queries_collection = self.db_client.get_collection('queries')
    docs_cursor = queries_collection.find() if self.take is None else queries_collection.find().limit(self.take) 

    for doc in docs_cursor:
      yield self.__map_from_db(doc)

  def search(self, github_url: str) -> Query | None:
    queries_collection = self.db_client.get_collection('queries')
    db_pair = queries_collection.find_one(filter={ "github_url": github_url })

    return None if db_pair is None else self.__map_from_db(db_pair)

  def get_dataset_count(self) -> int:
    if self.take is not None:
      return self.take
    
    queries_collection = self.db_client.get_collection('queries')
    return queries_collection.count_documents(filter={})
  
  def mark_as_not_found(self, queries: List[Query]):
    return self.db_client.get_collection('queries').update_many({ "github_url": { "$in": [query.url for query in queries] } }, { "$set": { "not_found": True }})
  
  def __map_from_db(self, doc: dict):
    return Query(
      url=doc['url'],
      language=doc['language'],
      query=doc['query'],
      relevance=doc['relevance']
    )
