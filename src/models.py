from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (Any, Callable, Dict, Generator, Generic, Iterable, List,
                    Literal, Optional, Tuple, TypedDict, TypeVar, Union)
from bson import ObjectId

import tensorflow as tf
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast # type: ignore
from transformers.modeling_tf_outputs import TFBaseModelOutput

Embedding = tf.Tensor
Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast
Partition = Literal['train', 'valid', 'test']
Language = Literal['python', 'java']
ModelInput = Tuple[tf.Tensor, tf.Tensor]

@dataclass
class Query:
  query: str
  relevance: int
  language: str
  url: str

@dataclass
class ValidationResult:
  query: Query
  code_url: str
  similarity: float

# TODO: Replace ValidationResult with this?
@dataclass
class Result:
  code_url: str
  relevance: int
  similarity: float

ItemType = TypeVar('ItemType')

class ResultAnalyzer(metaclass=ABCMeta):
  def print_results(self, results: List[ValidationResult]):
    total = len(results)
    correct_results = len([result for result in results if self.is_result_correct(result.similarity, result.query.relevance)])
    wrong_results = total - correct_results
    print(f"""
--------
Results count: {len(results)}
  correct: {correct_results}
  wrong: {wrong_results}
--------
""")

  @abstractmethod
  def is_result_correct(self, similarity: float, relevance: int) -> bool:
    raise NotImplementedError()

class EmbeddingComparator(metaclass=ABCMeta):
  name = ""

  @abstractmethod
  def fit(self, inputs: ModelInput, epochs: int, batch_count: int):
    raise NotImplementedError()

  @abstractmethod
  def save(self):
    raise NotImplementedError()
  
  @abstractmethod
  def load(self):
    raise NotImplementedError()

  @abstractmethod
  def predict(self, inputs) -> int:
    raise NotImplementedError()
  
  @abstractmethod
  def summary(self):
    raise NotImplementedError()
  
class DatasetRepository(Generic[ItemType], metaclass=ABCMeta):
  @abstractmethod
  def get_dataset(self) -> Iterable[ItemType]:
    raise NotImplementedError()
  
  @abstractmethod
  def search(self, github_url: str) -> Optional[ItemType]:
    raise NotImplementedError()
  
  @abstractmethod
  def get_dataset_count(self) -> int:
    raise NotImplementedError()
  
class PreProcesser(metaclass=ABCMeta):
  @abstractmethod
  def process_text(self, text_tokens: List[str]) -> str:
    raise NotImplementedError()
  
  @abstractmethod
  def process_code(self, code_tokens: List[str]) -> str:
    raise NotImplementedError()
  
class EmbeddingGenerator(metaclass=ABCMeta):
  @abstractmethod
  def from_text(self, text: str | list[str]) -> Embedding:
    raise NotImplementedError()
  
  @abstractmethod
  def from_code(self, code: str | list[str]) -> Embedding:
    raise NotImplementedError()
  
  @abstractmethod
  def target(self, target: int, batch_size: int) -> Embedding:
    raise NotImplementedError()
  
class EmbeddingConcat(metaclass=ABCMeta):
  @abstractmethod
  def concatenate(self, code_embedding: Embedding, text_embedding: Embedding, reshape: Optional[tuple] = None) -> Embedding:
    raise NotImplementedError()
  
class CodeSearchNetPair(TypedDict):
  url: str
  code_tokens: List[str]
  comment_tokens: List[str]
  partition: Partition
  language: Language

MongoId = ObjectId
class MongoDbPairDoc(TypedDict):
  _id: MongoId
  code_tokens: List[str]
  comment_tokens: List[str]
  github_url: str
  language: Language
  partition: Partition

class QueryRawDbDoc(TypedDict):
  _id: MongoId
  language: Language
  query: str
  relevance: int
  url: str

class QueryDbDoc(TypedDict):
  _id: MongoId
  language: Language
  query: str
  relevance: int
  url: str
  pair_doc: MongoDbPairDoc

# TODO: Review this!!!
# class CodeSearchNetPair(TypedDict):
#   _id: MongoId
#   language: Language
#   query: str
#   relevance: int
#   url: str
#   pair_doc: PairDbDoc


# TODO: Review the models below 👇

JsonData = dict[str, Any]


class JsonParser(metaclass=ABCMeta):
  @abstractmethod
  def from_json(self, json_data: Union[str, bytes]) -> JsonData:
      pass


class DatasetType(str, Enum):
  train = 'train'
  test = 'test'
  validation = 'valid'


class SupportedCodeLanguages(str, Enum):
  java = 'java'
  python = 'python'


EmbeddingModel = Callable[[], TFBaseModelOutput]
  
class Runnable(metaclass=ABCMeta):
  @abstractmethod
  def run(self):
    pass
