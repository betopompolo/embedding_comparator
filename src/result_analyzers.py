from typing import List
from models import Result
    
class ResultAnalyzerSigmoid:
  def is_result_correct(self, similarity: float, relevance: int):
    if 0 <= similarity <= 0.5:
      return relevance <= 1
    
    if 0.5 < similarity <= 1:
      return 2 <= relevance <= 3
  
  def print_results(self, results: List[Result]):
    total = len(results)
    correct_results = len([result for result in results if self.is_result_correct(result.similarity, result.relevance)])
    wrong_results = total - correct_results
    print(f"""
--------
Results count: {len(results)}
  correct: {correct_results}
  wrong: {wrong_results}
--------
""")
