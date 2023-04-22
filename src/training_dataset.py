from dataclasses import dataclass
import os
from typing import Iterable

import tensorflow as tf
from models import CodeCommentPair, DatasetRepository, JsonParser
from utils import decode_tensor_string, random_seed

dataset_len = {
  "test": {
    "java": 26880,
    "python": 22144
  },
  "train": {
    "java": 454451,
    "python": 412178
  },
  "valid": {
    "java": 15296,
    "python": 23104
  }
}
training_samples_count = sum(dataset_len['train'].values())

lines_count_in_json_file = 30000

@dataclass
class TrainingDataset(DatasetRepository):
  jsonParser: JsonParser
  samples_count: int = training_samples_count

  def get_dataset(self) -> Iterable[CodeCommentPair]:
    train_files = tf.data.Dataset.list_files(os.path.join(os.getcwd(), 'datasets', 'train', '*.jsonl'))
    for jsonl in tf.data.TextLineDataset(train_files).take(self.samples_count).shuffle(buffer_size=int(lines_count_in_json_file * 0.1), seed=random_seed):
      code_comment_pair = self.parse_jsonl(jsonl)
      yield code_comment_pair
  
  def parse_jsonl(self, string_tensor) -> CodeCommentPair:
    jsonl = self.jsonParser.from_json(decode_tensor_string(string_tensor))
    return CodeCommentPair(
      id=jsonl['url'],
      code_tokens=jsonl['code_tokens'],
      comment_tokens=jsonl['docstring_tokens'],
    )
