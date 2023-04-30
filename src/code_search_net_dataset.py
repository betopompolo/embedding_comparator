import os
from dataclasses import dataclass
from typing import Iterable

import tensorflow as tf

from models import CodeCommentPair, DatasetRepository, JsonParser
from utils import codesearchnet_dataset_len, decode_tensor_string


@dataclass
class CodeSearchNetDataset(DatasetRepository):
  json_parser: JsonParser

  def get_dataset(self) -> Iterable[CodeCommentPair]:
    all_files = tf.data.Dataset.list_files(os.path.join(os.getcwd(), 'datasets', '*', '*.jsonl'))
    for jsonl in tf.data.TextLineDataset(all_files):
      code_comment_pair = self.parse_jsonl(jsonl)
      yield code_comment_pair

  def get_dataset_count(self) -> int:
    count = 0
    for key in codesearchnet_dataset_len:
      count += sum(codesearchnet_dataset_len[key].values())
    return count
  
  def parse_jsonl(self, string_tensor) -> CodeCommentPair:
    jsonl = self.json_parser.from_json(decode_tensor_string(string_tensor))
    return CodeCommentPair(
      id=jsonl['url'],
      code_tokens=jsonl['code_tokens'],
      comment_tokens=jsonl['docstring_tokens'],
      partition=jsonl['partition']
    )
