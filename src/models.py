from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generator, Generic, Iterable, List, Literal, Optional, TypeVar, Union
import tensorflow as tf

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.modeling_tf_outputs import TFBaseModelOutput


Embedding = tf.Tensor
Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast
@dataclass
class CodeCommentPair:
  id: str
  code_tokens: List[str]
  comment_tokens: List[str]
  partition: Literal['train', 'valid', 'test']

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

ItemType = TypeVar('ItemType')
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
  def concatenate(self, code_embedding: Embedding, text_embedding: Embedding, reshape: Optional[tuple]) -> Embedding:
    raise NotImplementedError()

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

class CodeCommentPairRepository(metaclass=ABCMeta):
  @abstractmethod
  def batch(self, code_language: SupportedCodeLanguages, dataset_type: DatasetType, batch_size: int) -> Generator[List[CodeCommentPair], None, None]:
      pass
