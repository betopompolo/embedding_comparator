from dataclasses import dataclass
from typing import List, Literal

from models import Language


@dataclass
class ExperimentParameters:
  name: str
  num_hidden_layers: Literal[2] | Literal[4] | Literal[8]
  programming_languages: List[Language]
  

  def to_csv(self) -> str:
    return f'{self.num_hidden_layers},{",".join(self.programming_languages)}'
