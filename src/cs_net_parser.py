import csv
import os
from typing import Dict, Iterator, List, Literal, TypedDict, cast, get_args

import orjson
import tensorflow as tf

CSNetPartition = Literal['train', 'test', 'valid']
CSNetCodeLanguage = Literal['ruby', 'go', 'java', 'javascript', 'php', 'python']

class CSNetSample(TypedDict):
  repo: str
  path: str
  func_name: str
  original_string: str
  language: CSNetCodeLanguage
  code: str
  code_tokens: List[str]
  docstring: str
  docstring_tokens: List[str]
  sha: str
  url: str
  partition: CSNetPartition

CSNetQueryLanguage = Literal[ 'Ruby', 'Go', 'Java', 'JavaScript', 'PHP', 'Python']
class CSNetQuery(TypedDict):
  Language: CSNetQueryLanguage
  Query: str
  GitHubUrl: str
  Relevance: int
  Notes: str


class CSNetParser:
  __valid_query_languages = get_args(CSNetQueryLanguage)
  __dataset_len_info: Dict[CSNetPartition, Dict[CSNetCodeLanguage, int]] = {
    "test": {
      "python": 22176
    },
    "train": {
      "python": 412178
    },
    "valid": {
      "python": 23107
    }
  }
  
  def get_samples(self, partition: CSNetPartition = 'train', language: CSNetCodeLanguage = 'python') -> Iterator[CSNetSample]:
    dataset_dir = os.path.join(os.path.abspath(os.curdir), f'datasets/temp_{language}', language, 'final', 'jsonl', partition)
    file_names = [os.path.join(dataset_dir, file_name) for file_name in os.listdir(dataset_dir) if file_name.endswith('.jsonl.gz')]

    tf_dataset = tf.data.TextLineDataset(
      filenames=file_names,
      compression_type='GZIP',
      num_parallel_reads=tf.data.AUTOTUNE,
    )

    for tensor in tf_dataset:
      yield orjson.loads(tensor.numpy()) # type: ignore

  def get_queries(self) -> Iterator[CSNetQuery]:
    queries_path = os.path.join(os.path.abspath(os.curdir), 'datasets/code_search_net_queries.csv')

    with open(queries_path, encoding='utf-8') as csvf:
      csvReader = csv.DictReader(csvf)
      for row in csvReader:
        query: CSNetQuery = {
          'GitHubUrl': row['GitHubUrl'],
          'Language': self.parse_query_language(row['Language']),
          'Relevance': int(row['Relevance']),
          'Query': row['Query'],
          'Notes': row['Notes'],
        }

        yield query

  def get_dataset_len(self, partition: CSNetPartition, language: CSNetCodeLanguage) -> int:
    return self.__dataset_len_info[partition][language]
  
  def parse_query_language(self, data: str) -> CSNetQueryLanguage:
    assert data in self.__valid_query_languages, f"Query language got value {data}, but should be one of these: {self.__valid_query_languages}"
    return cast(CSNetQueryLanguage, data)
