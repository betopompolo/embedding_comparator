import os
import pickle
from typing import Iterator, TypedDict

import numpy as np

from embedding_generator import EmbeddingPairBatch

class EmbeddingPair(TypedDict):
  id: str
  code_embedding: np.ndarray
  comment_embedding: np.ndarray


class EmbeddingDataset:
  def __init__(self, dataset_name: str, dataset_dir: str = '') -> None:
    dataset_extension = '.pkl'
    self.dataset_name = dataset_name if dataset_name.endswith(dataset_extension) else f'{dataset_name}{dataset_extension}'
    self.dataset_dir = dataset_dir if len(dataset_dir) > 0 else os.path.join(os.path.abspath(os.curdir), 'datasets/embeddings')

  def save(self, embeddings: EmbeddingPairBatch):
    with open(self.get_dataset_full_path(), 'ab') as file:
      for pair_id, code_embedding, comment_embedding in zip(embeddings['pairs_ids'], embeddings['code_embeddings'], embeddings['comment_embeddings']):
        embedding_pair: EmbeddingPair = {
          "id": pair_id,
          "code_embedding": code_embedding,
          "comment_embedding": comment_embedding,
        }
        pickle.dump(embedding_pair, file)

  def get(self, pair_id: str) -> EmbeddingPair | None:
    for embedding in self.list_all():
      if embedding['id'] == pair_id:
        return embedding

  def list_all(self) -> Iterator[EmbeddingPair]:
    with open(self.get_dataset_full_path(), 'rb') as file:
      while True:
        try:
          yield pickle.load(file)
        except EOFError:
          break

      
  def get_dataset_full_path(self):
    return os.path.join(self.dataset_dir, self.dataset_name)
