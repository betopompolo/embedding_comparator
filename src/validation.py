from dataclasses import dataclass
from datetime import datetime
from typing import List

from tqdm import tqdm

from models import (CodeCommentPair, DatasetRepository, EmbeddingComparator, EmbeddingConcat,
                    EmbeddingGenerator, PreProcesser, Query, ResultAnalyzer, ValidationResult)


@dataclass
class Validation:
  cs_net_repository: DatasetRepository[CodeCommentPair]
  query_repository: DatasetRepository[Query]
  embedding_concat: EmbeddingConcat
  embedding_generator: EmbeddingGenerator
  pre_processer: PreProcesser
  model: EmbeddingComparator
  result_analyzer: ResultAnalyzer

  def run(self):
    self.model.load()
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

    self.write_results(results, file_name_suffix=self.model.name)
    self.result_analyzer.print_results(results)

  def write_results(self, results: List[ValidationResult], file_name_suffix: str = "result"):
    with open(f'results/{datetime.now().isoformat()}-{file_name_suffix}.csv', 'w') as file:
      for result in tqdm(results, desc='Writing results'):
        file.write(f'{result.similarity}, {result.query.relevance}\n')
