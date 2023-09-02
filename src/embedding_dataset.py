from contextlib import AbstractContextManager
import os
from types import TracebackType
from typing import Literal, TypedDict, cast
import h5py
from h5py import Group
import numpy as np

from embedding_generator import EmbeddingPairBatch

class EmbeddingPair(TypedDict):
  id: str
  code_embedding: np.ndarray
  comment_embedding: np.ndarray


class EmbeddingDataset(AbstractContextManager):
  def __init__(self, dataset_name: str, dataset_dir: str = '', mode: Literal['r', 'w'] = 'r') -> None:
    dataset_extension = '.hdf5'
    self.dataset_name = dataset_name if dataset_name.endswith(dataset_extension) else f'{dataset_name}{dataset_extension}'
    self.dataset_dir = dataset_dir if len(dataset_dir) > 0 else os.path.join(os.path.abspath(os.curdir), 'datasets/embeddings')
    self.code_embeddings_group = 'code_embeddings'
    self.comment_embeddings_group = 'comment_embeddings'
    if mode == 'w':
      self.mode = 'r+' if os.path.isfile(self.get_dataset_full_path()) else mode
    else:
      self.mode = mode

  def __enter__(self):
    self.db = h5py.File(self.get_dataset_full_path(), mode=self.mode)
    return self

  def __exit__(self, __exc_type: type[BaseException] | None, __exc_value: BaseException | None, __traceback: TracebackType | None) -> bool | None:
    if self.db is not None:
      self.db.close()
    
    return super().__exit__(__exc_type, __exc_value, __traceback)

  def save(self, embeddings: EmbeddingPairBatch):
    self.__assert_db()
    code_embeddings_group = self.__get_db_group(self.code_embeddings_group)
    comments_embeddings_group = self.__get_db_group(self.comment_embeddings_group)

    for pair_id, code_embedding, comment_embedding in zip(embeddings['pairs_ids'], embeddings['code_embeddings'], embeddings['comment_embeddings']):
      code_embeddings_group[pair_id] = code_embedding # type: ignore
      comments_embeddings_group[pair_id] = comment_embedding # type: ignore

  def get(self, pair_id: str) -> EmbeddingPair | None:
    self.__assert_db()
    code_embeddings_group = self.__get_db_group(self.code_embeddings_group)
    comments_embeddings_group = self.__get_db_group(self.comment_embeddings_group)

    code_embedding = code_embeddings_group.get(pair_id)
    comment_embedding = comments_embeddings_group.get(pair_id)

    if code_embedding is None or comment_embedding is None:
      return None
    
    return {
      'id': pair_id,
      'code_embedding': code_embedding[()], # type: ignore
      'comment_embedding': comment_embedding[()], # type: ignore
    }

      
  def get_dataset_full_path(self):
    return os.path.join(self.dataset_dir, self.dataset_name)
  
  def __assert_db(self):
    assert hasattr(self, 'db'), "EmbeddingDataset should be used alongside with python's 'with' keyword"

  def __get_db_group(self, group_name: str) -> Group:
    if group_name in self.db:
      return cast(Group, self.db[group_name])
    else:
      return self.db.create_group(group_name)
