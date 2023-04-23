from typing import Iterable

import tensorflow as tf
from tqdm import tqdm
from models import DatasetRepository, Query
from utils import decode_tensor_string


class QueryDataset(DatasetRepository[Query]):
  def get_dataset(self) -> Iterable[Query]:
    query_lines = tf.data.TextLineDataset('datasets/queries.csv')
    for tensor_csv in query_lines:
      yield self.__map_csv(decode_tensor_string(tensor_csv))
  
  def get_dataset_count(self) -> int:
    return 2894

  def __map_csv(self, csv: str) -> Query:
    values = csv.split(',')
    [language, query, github_url, relevance, *notes] = values

    return Query(
      language=language,
      query=query,
      url=github_url,
      relevance=int(relevance)
    )

ds = QueryDataset()
for q in tqdm(ds.get_dataset(), total=ds.get_dataset_count()):
  pass