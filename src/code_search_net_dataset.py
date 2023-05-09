from dataclasses import dataclass
import os
from typing import Iterable, List

import tensorflow as tf

from models import CodeCommentPair, DatasetRepository, JsonParser, Language, Partition
from utils import codesearchnet_dataset_len, decode_tensor_string


@dataclass
class CodeSearchNetDataset(DatasetRepository[CodeCommentPair]):
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
  