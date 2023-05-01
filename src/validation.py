import itertools
from dataclasses import dataclass
from typing import Dict, List

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
    results: Dict[str, List[ValidationResult]] = {}
    not_found_urls = []
    
    for query in tqdm(queries_dataset, total=self.query_repository.get_dataset_count()):
      pair = self.cs_net_repository.search(query.url)
      if pair is None:
        not_found_urls.append(query.url)
        continue

      embedding_query = self.embedding_generator.from_text(
        self.pre_processer.process_text(query.query.split())
      )
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

    print('all queries were found!') if len(not_found_urls) == 0 else print(f'{len(not_found_urls)} queries were not found in code_search_net database')
    
    # TODO: Move!
    # def print_query_results(search_term: str, results: List[ValidationResult]):
    #   print(f'search_term: {search_term}')
    #   for result in results:
    #     print(f'code: {result.code_url} similarity: {result.similarity}')
    #   print(f'----------------')
    
    # for search_term in results.keys():
    #   results[search_term].sort(key=lambda result: result.similarity, reverse=True)
    #   print_query_results(search_term, results[search_term])
    #   break


    
Validation(
  cs_net_repository=CodeSearchNetDataset(),
  query_repository=QueryDataset(),
  embedding_concat=EmbeddingConcatDefault(),
  embedding_generator=EmbeddingGeneratorDefault(),
  model=EmbeddingComparator(),
  pre_processer=PreProcesserDefault()
).run()
