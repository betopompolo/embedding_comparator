from dataclasses import dataclass
from typing import List

import tensorflow as tf
from embedding_generator_default import EmbeddingGeneratorDefault
from experiment_parameters import ExperimentParameters

from models import EmbeddingGenerator, MongoId, MongoDbPairDoc, Runnable
from mongo_db_client import MongoDbClient
from pymongo.collection import Collection

from pre_processer_default import PreProcesserDefault
from utils import build_model, get_model_path, encoder_hidden_size, encoder_seq_len


@dataclass
class Training(Runnable):
  experiments: List[ExperimentParameters]
  train_samples = 100 # TODO: Change this parameter
  num_epochs = 10
  batch_size = 1
  db_client = MongoDbClient()

  def run(self):   
    for experiment_parameters in self.experiments:
      training_model = build_model(experiment_parameters.num_hidden_layers)
      embedding_generator = EmbeddingGeneratorDefault(
        pre_processer=PreProcesserDefault()
      )
      
      embedding_spec = tf.TensorSpec(shape=(self.batch_size, encoder_seq_len, encoder_hidden_size), dtype=tf.float64) # type: ignore
      target_spec = tf.TensorSpec(shape=(self.batch_size, ), dtype=tf.int32) # type: ignore
      dataset = tf.data.Dataset.from_generator(
        lambda : self.generate_dataset(experiment_parameters, embedding_generator),
        output_signature=((embedding_spec, embedding_spec), target_spec),
      ).shuffle(buffer_size=int(self.train_samples * 0.1))

      training_model.fit(dataset, epochs=self.num_epochs, steps_per_epoch=self.train_samples/self.num_epochs)
      training_model.save(get_model_path(experiment_parameters))

  def generate_dataset(self, experiment_parameters: ExperimentParameters, embedding_generator: EmbeddingGenerator):
    pairs = self.db_client.get_pairs_collection()
    pairs_query = {
      "language": {
        "$in": experiment_parameters.programming_languages
      },
      "partition": "train",
    }

    for pair_doc in pairs.find(pairs_query).limit(int(self.train_samples / 2)):
      target_shape = (self.batch_size, )
      code_emb, comment_emb, target = embedding_generator.from_code(pair_doc['code_tokens']), embedding_generator.from_text(pair_doc['comment_tokens']), tf.ones(shape=target_shape)
      yield ((code_emb, comment_emb), target)

      negative_doc = self.random_pair(pairs, exclude_id=pair_doc['_id'])
      negative_code_emb, negative_comment_emb, negative_target = embedding_generator.from_code(negative_doc['code_tokens']), embedding_generator.from_text(pair_doc['comment_tokens']), tf.zeros(shape=target_shape)
      yield ((negative_code_emb, negative_comment_emb), negative_target)
  
  def random_pair(self, pairs_collection: Collection[MongoDbPairDoc], exclude_id: MongoId) -> MongoDbPairDoc:
    return pairs_collection.aggregate([
      {'$sample': {'size': 1}},
      {'$match': {
        "$and": [
          { '_id': {'$ne': exclude_id} },
          { 'partition': "train" },
        ],
      }}
    ]).next()