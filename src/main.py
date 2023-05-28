from tqdm import tqdm
from local_datasets import TrainingPairsRepository, QueriesRepository, ValidationPairsRepository
from embedding_comparator_dense import EmbeddingComparatorDense
from embedding_concat_default import EmbeddingConcatDefault
from embedding_generator_default import EmbeddingGeneratorDefault
from pre_processer_default import PreProcesserDefault
from result_analyzers import ResultAnalyzerBinary
from training import Training
from validation import Validation

batch_size = 128
training_samples_count = batch_size * 5
negative_samples_count = 1

model = EmbeddingComparatorDense(name=f"dense_{training_samples_count}", batch_size=batch_size)
embedding_concat=EmbeddingConcatDefault()
embedding_generator=EmbeddingGeneratorDefault()
pre_processer=PreProcesserDefault()

# Training(
#   pre_processer=pre_processer,
#   model=model,
#   embedding_concat=embedding_concat,
#   embedding_generator=embedding_generator,
#   batch_size=batch_size,
#   training_samples_count=training_samples_count,
#   negative_samples_count=negative_samples_count, 
#   dataset_repository=TrainingPairsRepository(
#     take=int(training_samples_count / (negative_samples_count + 1))
#   ),
# ).run()

# Validation(
#   embedding_concat=embedding_concat,
#   embedding_generator=embedding_generator,
#   model=model,
#   pre_processer=pre_processer,
#   cs_net_repository=ValidationPairsRepository(),
#   query_repository=QueriesRepository(take=500),
#   result_analyzer=ResultAnalyzerBinary(),
# ).run()

queries_repo = QueriesRepository()
all_pairs_repo = ValidationPairsRepository()
not_found_queries = []

for query in tqdm(queries_repo.get_dataset(), total=queries_repo.get_dataset_count(), desc="Finding queries that aren't in the cs net dataset"):
  if all_pairs_repo.search(query.url) is None:
    not_found_queries.append(query)
  
result = queries_repo.mark_as_not_found(not_found_queries)
print(f'{result.modified_count} queries were modified')
