from dataclasses import dataclass
from typing import List, cast

from tqdm import tqdm
from embedding_generator_default import EmbeddingGeneratorDefault
from experiment_parameters import ExperimentParameters
from models import Result, Runnable

import tensorflow as tf
from keras import Model
from mongo_db_client import MongoDbClient
from pre_processer_default import PreProcesserDefault

from utils import get_model_path

@dataclass
class QueryValidation(Runnable):
  db_client = MongoDbClient()
  experiments: List[ExperimentParameters]

  def run(self):
    embedding_generator = EmbeddingGeneratorDefault(
      pre_processer=PreProcesserDefault()
    )
    queries_collection = self.db_client.get_queries_collection()

    for experiment_parameters in self.experiments:
      results = []
      validation_model = self.load_model(experiment_parameters)
      queries = queries_collection.find({ "language": { "$in": experiment_parameters.programming_languages } })
      for query_doc in tqdm(queries, total=queries_collection.count_documents({}), desc=f"Validating {experiment_parameters.name}"):
        pair_doc = query_doc['pair_doc']
        code_emb, comment_emb = embedding_generator.from_code(
          pair_doc['code_tokens']), embedding_generator.from_text(query_doc['query'].split())
        prediction = validation_model.predict(
          x=[code_emb, comment_emb], verbose="0")
        similarity = prediction.item()

        results.append(Result(
          code_url=query_doc['url'],
          relevance=query_doc['relevance'],
          similarity=similarity,
        ))

  def load_model(self, experiment_parameters: ExperimentParameters):
    return cast(Model, tf.keras.models.load_model(get_model_path(experiment_parameters)))
  
  def write_results(self, experiment: ExperimentParameters, result_list: List[Result]):
    with open(f'results/{experiment.name}.csv', 'w') as file:
      for result in tqdm(result_list, desc='Writing results'):
        file.write(f'{result.similarity}, {result.relevance}\n')