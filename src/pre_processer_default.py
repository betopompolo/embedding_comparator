from typing import List
from models import PreProcesser


class PreProcesserDefault(PreProcesser):
  def process_text(self, text_tokens: List[str]) -> str:
    parsed=' '.join(text_tokens).replace('\n',' ')
    parsed=' '.join(parsed.strip().split())
    return parsed

  def process_code(self, code_tokens: List[str]) -> str:
    parsed=' '.join(code_tokens).replace('\n',' ')
    parsed=' '.join(parsed.strip().split())
    return parsed
