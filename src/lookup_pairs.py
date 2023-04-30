from typing import Dict, List
import pandas as pd

from models import CodeCommentPair

class SearchPair:
  lookup_in_memory: Dict[str, List[str]]

  def search(self, pair_id: str) -> CodeCommentPair:
    
