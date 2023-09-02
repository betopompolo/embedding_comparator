from runnable import Runnable

"""
TODO:
- Load queries from cs net queries
- For each embedding stored (es) in embedding dataset
  - For each query (q)
    - get code embedding from (es) and comment_embedding from (query)
    - predict using model.predict
"""
class QueryValidation(Runnable):
  def run(self):
    