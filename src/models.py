from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generator, Iterable, List, Union
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
class DatasetRepository(metaclass=ABCMeta):
  @abstractmethod
  def get_dataset(self) -> Iterable[CodeCommentPair]:
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
  def concatenate(self, code_embedding: Embedding, text_embedding: Embedding) -> Embedding:
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
