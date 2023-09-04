from datetime import datetime
import os
from typing import Iterator
import more_itertools
import numpy as np
import tensorflow as tf

from embedding_dataset import EmbeddingDataset
from embedding_generator import EmbeddingGenerator
from mongo_db_client import MongoDbClient, MongoDbQueryDoc
from runnable import Runnable
from keras.models import load_model, Model


class CSNetValidation(Runnable):
  def __init__(self, model_name: str) -> None:
    self.model_path = os.path.join(os.path.abspath(os.curdir), 'models', model_name)

  def run(self):
    model: Model | None = load_model(self.model_path)
    assert model is not None, f'Model at path {self.model_path} was not found'
    
    mongo_db = MongoDbClient()
    embedding_generator = EmbeddingGenerator()
    queries = mongo_db.get_queries_collection().find().limit(1) # TODO: remove limit (or change 1)
    batch_size = 100

    tf_dataset = self.create_tf_dataset(embedding_generator, queries).batch(batch_size)
    predicts = model.predict(tf_dataset)

    run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    np.save(os.path.join(os.path.abspath(os.curdir), 'results', run_timestamp), predicts)

  def create_tf_dataset(self, embedding_generator: EmbeddingGenerator, queries: Iterator[MongoDbQueryDoc]):
    def generator():
      with EmbeddingDataset(dataset_name='embeddings') as embedding_dataset:
        for batch_queries in more_itertools.chunked(queries, 200):
          query_embeddings = embedding_generator.from_sentences(
            [query['query'] for query in batch_queries], 
            tokenizer=embedding_generator.comment_embedding_tokenizer, 
            model=embedding_generator.comment_embedding_model,
          ).numpy()

          for query_embedding in query_embeddings:
            for pair_embedding in embedding_dataset.list():
              yield {
                "code_embedding": pair_embedding['code_embedding'],
                "comment_embedding": query_embedding,
              }

    return tf.data.Dataset.from_generator(generator, output_types={
      "code_embedding": tf.float32, 
      "comment_embedding": tf.float32,
    })
