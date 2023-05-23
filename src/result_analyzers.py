from models import ResultAnalyzer


class ResultAnalyzerBinary(ResultAnalyzer):
  def is_result_correct(self, similarity: int, relevance: int):
    if similarity == 0:
      return relevance <= 1
    else:
      return 2 <= relevance <= 3
