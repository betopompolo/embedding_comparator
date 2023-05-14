from dataclasses import dataclass
from datetime import datetime
import os
from typing import List

from tqdm import tqdm

from code_search_net_local_repository import CodeSearchNetLocalRepository

from embedding_comparator_dense import EmbeddingComparatorDense
from embedding_concat_default import EmbeddingConcatDefault
from embedding_generator_default import EmbeddingGeneratorDefault
from models import (CodeCommentPair, DatasetRepository, EmbeddingComparator, EmbeddingConcat,
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
    model_name = 'dense_128000'
    self.model.load(model_name)
    queries_dataset = self.query_repository.get_dataset()
    results: List[ValidationResult] = []
    not_found_count = 0

    for query in tqdm(queries_dataset, total=self.query_repository.get_dataset_count()):
      pair = self.cs_net_repository.search(query.url)
      if pair is None:
          not_found_count += 1
          continue

      embedding_query = self.embedding_generator.from_text(
          self.pre_processer.process_text(query.query.split())
      ) 
      embedding_code = self.embedding_generator.from_code(
          self.pre_processer.process_code(pair.code_tokens))
      concatenated = self.embedding_concat.concatenate(
          embedding_code, embedding_query, reshape=(1, -1))
      similarity = self.model.predict(concatenated)
      result = ValidationResult(
        query=query,
        code_url=pair.id,
        similarity=similarity,
      )
      results.append(result)

    self.write_results(results, file_name_sufix=model_name)

  def write_results(self, results: List[ValidationResult], file_name_sufix: str = "result"):
    with open(f'results/{datetime.now().isoformat()}-{file_name_sufix}.csv', 'w') as file:
      for result in tqdm(results, desc='Writing results'):
        file.write(f'{result.similarity}, {result.query.relevance}\n')

  def is_result_correct(self, similarity: float, relevance: int):
    ranges = [[0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1]]
    similarity_range = ranges[relevance]
    return similarity_range[0] <= similarity <= similarity_range[1]


Validation(
  cs_net_repository=CodeSearchNetLocalRepository(),
  query_repository=QueryDataset(),
  embedding_concat=EmbeddingConcatDefault(),
  embedding_generator=EmbeddingGeneratorDefault(),
  model=EmbeddingComparatorDense(),
  pre_processer=PreProcesserDefault()
).run()
