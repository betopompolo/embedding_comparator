import itertools
from dataclasses import dataclass
from typing import Dict, List

from tqdm import tqdm

from embedding_comparator import EmbeddingComparator
from embedding_concat_default import EmbeddingConcatDefault
from embedding_generator_default import EmbeddingGeneratorDefault
from json_parser import OrJsonParser
from models import (CodeCommentPair, DatasetRepository, EmbeddingConcat,
                    EmbeddingGenerator, PreProcesser, Query, ValidationResult)
from pre_processer_default import PreProcesserDefault
from query_dataset import QueryDataset
from validation_dataset import ValidationDataset


@dataclass
class Validation:
  validation_repository: DatasetRepository[CodeCommentPair]
  queries_dataset: DatasetRepository[Query]
  embedding_concat: EmbeddingConcat
  embedding_generator: EmbeddingGenerator
  pre_processer: PreProcesser
  model: EmbeddingComparator

  def run(self):
    validation_dataset = self.validation_repository.get_dataset()
    queries_dataset = self.queries_dataset.get_dataset()
    results: Dict[str, List[ValidationResult]] = {}

    test_slice_count = 50 # TODO: remove!!!
    
    with tqdm(total=test_slice_count) as progress_bar:
      for query in itertools.islice(queries_dataset, 1):
        embedding_query = self.embedding_generator.from_text(
          self.pre_processer.process_text(query.query.split())
        )
        for pair in itertools.islice(validation_dataset, test_slice_count):
          embedding_code = self.embedding_generator.from_code(self.pre_processer.process_code(pair.code_tokens))
          concatenated = self.embedding_concat.concatenate(embedding_code, embedding_query, reshape=(1, -1))
          similarity = self.model.predict(concatenated, hide_logs=True)[:1]
          result = ValidationResult(
            query=query,
            code_url=pair.id,
            similarity=similarity,
          )

          if query.query in results:
            results[query.query].append(result)
          else:
            results[query.query] = [result]
          progress_bar.update(1)
    
    # TODO: Move!
    def print_query_results(search_term: str, results: List[ValidationResult]):
      print(f'search_term: {search_term}')
      for result in results:
        print(f'code: {result.code_url} similarity: {result.similarity}')
      print(f'----------------')
    
    for search_term in results.keys():
      results[search_term].sort(key=lambda result: result.similarity, reverse=True)
      print_query_results(search_term, results[search_term])
      break


    
Validation(
  validation_repository=ValidationDataset(
    json_parser=OrJsonParser(),
  ),
  queries_dataset=QueryDataset(),
  embedding_concat=EmbeddingConcatDefault(),
  embedding_generator=EmbeddingGeneratorDefault(),
  model=EmbeddingComparator(),
  pre_processer=PreProcesserDefault()
).run()

