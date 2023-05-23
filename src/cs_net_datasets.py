from dataclasses import dataclass
import os
from typing import Iterable, List

import tensorflow as tf

from models import CodeCommentPair, DatasetRepository, JsonParser, Partition, Query
from utils import codesearchnet_dataset_len, decode_tensor_string


@dataclass
class CSNetPairDataset(DatasetRepository[CodeCommentPair]):
  json_parser: JsonParser

  def get_dataset(self) -> Iterable[CodeCommentPair]:
    partitions: List[Partition] = ['train', 'test', 'valid']

    for partition in partitions:
      java_files = tf.data.Dataset.list_files(os.path.join(os.getcwd(), 'datasets', partition, 'java_*.jsonl'))
      python_files = tf.data.Dataset.list_files(os.path.join(os.getcwd(), 'datasets', partition, 'python_*.jsonl'))

      for java_tensor in tf.data.TextLineDataset(java_files):
        yield self.parse_jsonl(java_tensor)

      for python_tensor in tf.data.TextLineDataset(python_files):
        yield self.parse_jsonl(python_tensor)

  def get_dataset_count(self) -> int:
    return sum(codesearchnet_dataset_len['train'].values()) + sum(codesearchnet_dataset_len['test'].values()) + sum(codesearchnet_dataset_len['valid'].values())
  
  def search(self, github_url: str) -> CodeCommentPair | None:
    return super().search(github_url)
  
  def parse_jsonl(self, string_tensor) -> CodeCommentPair:
    jsonl = self.json_parser.from_json(decode_tensor_string(string_tensor))
    return CodeCommentPair(
      id=jsonl['url'],
      code_tokens=jsonl['code_tokens'],
      comment_tokens=jsonl['docstring_tokens'],
      partition=jsonl['partition'],
      language=jsonl['language']
    )
  
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