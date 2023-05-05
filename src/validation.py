from dataclasses import dataclass
from typing import List

from tqdm import tqdm

from code_search_net_dataset import CodeSearchNetDataset

from embedding_comparator import EmbeddingComparator
from embedding_concat_default import EmbeddingConcatDefault
from embedding_generator_default import EmbeddingGeneratorDefault
from models import (CodeCommentPair, DatasetRepository, EmbeddingConcat,
                    EmbeddingGenerator, PreProcesser, Query, ValidationResult)
from pre_processer_default import PreProcesserDefault
from query_dataset import QueryDataset


@dataclass
class Validation:
  cs_net_repository: DatasetRepository[CodeCommentPair]
  query_repository: DatasetRepository[Query]
  embedding_concat: EmbeddingConcat
  embedding_generator: EmbeddingGenerator
  pre_processer: PreProcesser
  model: EmbeddingComparator

  def run(self):
    queries_dataset = self.query_repository.get_dataset()
    results: List[ValidationResult] = []
    not_found_count = 0
    success_count = 0
    error_count = 0
    
    for query in tqdm(queries_dataset, total=self.query_repository.get_dataset_count()):
      pair = self.cs_net_repository.search(query.url)
      if pair is None:
        not_found_count += 1
        continue
      
      
      embedding_query = self.embedding_generator.from_text(
        self.pre_processer.process_text(query.query.split())
      )
      embedding_code = self.embedding_generator.from_code(self.pre_processer.process_code(pair.code_tokens))
      concatenated = self.embedding_concat.concatenate(embedding_code, embedding_query, reshape=(1, -1))
      prediction = self.model.predict(concatenated, hide_logs=True)
      similarity = prediction[0][0]
      result = ValidationResult(
        query=query,
        code_url=pair.id,
        similarity=similarity,
      )
      results.append(result)
    
    for result in results:
      if self.is_result_correct(result):
        success_count += 1
      else:
        error_count += 1
    print('Summary')
    print(f'--------')
    print(f'num_queries: {self.query_repository.get_dataset_count()}')
    print(f'not found queries: {not_found_count}')
    print(f'success: {success_count}')
    print(f'error: {error_count}')
    print(f'--------')

  def is_result_correct(self, result: ValidationResult):
    ranges = [[0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1]]

    current_range = ranges[result.query.relevance]

    return current_range[0] <= result.query.relevance < current_range[1]

Validation(
  cs_net_repository=CodeSearchNetDataset(),
  query_repository=QueryDataset(),
  embedding_concat=EmbeddingConcatDefault(),
  embedding_generator=EmbeddingGeneratorDefault(),
  model=EmbeddingComparator(),
  pre_processer=PreProcesserDefault()
).run()
