# TODO: Remove this file
from dataclasses import dataclass
import os
from typing import Iterable, List

import tensorflow as tf

from models import DatasetRepository, JsonParser, Partition, Query
from utils import decode_tensor_string
  
class CSNetQueryDataset(DatasetRepository[Query]):
  def get_dataset(self) -> Iterable[Query]:
    query_lines = tf.data.TextLineDataset('datasets/queries.csv')
    for tensor_csv in query_lines:
      yield self.__map_csv(decode_tensor_string(tensor_csv))
  
  def get_dataset_count(self) -> int:
    return 2892
  
  def search(self, github_url: str) -> Query | None:
    return super().search(github_url)

  def __map_csv(self, csv: str) -> Query:
    values = csv.split(',')
    [language, query, github_url, relevance, *notes] = values

    return Query(
      language=language,
      query=query,
      url=github_url,
      relevance=int(relevance)
    )