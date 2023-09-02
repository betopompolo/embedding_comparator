from typing import List, TypedDict

from tqdm import tqdm
from cs_net_parser import CSNetCodeLanguage, CSNetPartition
from embedding_dataset import EmbeddingDataset
from embedding_generator import EmbeddingGenerator
from mongo_db_client import MongoDbClient
from runnable import Runnable


class CreateEmbeddingDbFilter(TypedDict):
   partition: CSNetPartition
   language: CSNetCodeLanguage
   count: int

class CreateEmbeddingDb(Runnable):
  def __init__(self, filters: List[CreateEmbeddingDbFilter]) -> None:
    self.filters = filters

  def run(self):
    mongo_db = MongoDbClient()
    embedding_generator = EmbeddingGenerator()

    for filter in self.filters:
      pairs = list(mongo_db.get_pairs_collection().find({ "partition": filter['partition'], "language": filter['language'] }).limit(filter["count"]))
      with tqdm(desc=f"Saving {filter['language']} embeddings from {filter['partition']} partition", total=len(pairs)) as progress_bar:
        with EmbeddingDataset('embeddings', mode='w') as embedding_db:
          for embeddings in embedding_generator.from_pairs(pairs):
            embedding_db.save(embeddings)
            progress_bar.update(len(embeddings['pairs_ids']))
          