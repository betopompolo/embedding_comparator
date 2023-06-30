from typing import Dict, Iterable, Literal, cast

import pandas as pd

from models import CodeSearchNetQuery

CodeSearchNetQueryProgrammingLanguage = Literal['Go', 'Java', 'JavaScript', 'PHP', 'Python', 'Ruby']
dataset_len: Dict[CodeSearchNetQueryProgrammingLanguage, int] = {
  'Java': 813,
  'Go': 166,
  'JavaScript': 319,
  'PHP': 314,
  'Python': 2079,
  'Ruby': 315,
}

class CodeSearchNetQueriesRepository:
  def read_queries(self, language: CodeSearchNetQueryProgrammingLanguage) -> Iterable[CodeSearchNetQuery]:
    queries_df = pd.read_csv('datasets/code_search_net_queries.csv', names=['Language','Query','GitHubUrl','Relevance','Notes'])

    for index, line in queries_df.iterrows():
      if index == 0:
        continue
      
      query = cast(CodeSearchNetQuery, line)
      if query['Language'] == language:
        yield query

  def get_count(self, language: CodeSearchNetQueryProgrammingLanguage) -> int:
    return dataset_len[language]
