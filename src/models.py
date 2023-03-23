from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generator, List, Union

from transformers.modeling_tf_outputs import TFBaseModelOutput


@dataclass
class CodeCommentPair:
    id: str
    code: str
    comment: str


@dataclass
class Experiment(metaclass=ABCMeta):
    @abstractmethod
    async def run(self):
        pass


JsonData = dict[str, str]


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
