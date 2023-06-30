from dataclasses import dataclass
import os
from typing import Dict, Iterator, Literal
import zipfile
import tensorflow as tf


from models import CodeSearchNetPair, JsonParser
from orjson_parser import OrJsonParser
from utils import decode_tensor_string

CodeSearchNetPartition = Literal['train', 'test', 'valid']
CodeSearchNetProgrammingLanguage = Literal['ruby', 'go', 'java', 'javascript', 'php', 'python']

dataset_len_info: Dict[CodeSearchNetPartition, Dict[CodeSearchNetProgrammingLanguage, int]] = {
  "test": {
    "java": 26909,
    "python": 22176
  },
  "train": {
    "java": 454451,
    "python": 412178
  },
  "valid": {
    "java": 15328,
    "python": 23107
  }
}


@dataclass
class CodeSearchNetRepository:
  json_parser: JsonParser = OrJsonParser()

  def read_dataset(self, language: CodeSearchNetProgrammingLanguage, partition: CodeSearchNetPartition) -> Iterator[CodeSearchNetPair]:
    extract_dir = f'datasets/temp_{language}'

    if os.path.isdir(extract_dir) == False:
      print(f'Extracting {language} dataset to {extract_dir} directory...')
      dataset_path = f'datasets/code_search_net/data/{language}.zip'
      with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
      print(f'Extracting {language} dataset was successfull')

    dataset_dir = os.path.join(extract_dir, language, 'final', 'jsonl', partition)
    names = [os.path.join(dataset_dir, file_name) for file_name in os.listdir(dataset_dir) if file_name.endswith('.jsonl.gz')]
    for tensor in tf.data.TextLineDataset(names, compression_type='GZIP'):
      jsonl = self.json_parser.from_json(decode_tensor_string(tensor))
      yield {
        'url': jsonl['url'],
        'code_tokens': jsonl['code_tokens'],
        'comment_tokens': jsonl['docstring_tokens'],
        'partition': jsonl['partition'],
        'language': jsonl['language']
      }

  def get_count_for(self, language: CodeSearchNetProgrammingLanguage, partition: CodeSearchNetPartition) -> int:
    return dataset_len_info[partition][language]