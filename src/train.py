import os
import tensorflow as tf
from datetime import datetime
from typing import Iterator, List
from keras import Model, callbacks
from embedding_dataset import EmbeddingDataset
from mongo_db_client import MongoDbClient, MongoDbPairDoc
from runnable import Runnable


class Train(Runnable):
  def __init__(self, model: Model, train_count: int, valid_count: int) -> None:
    self.model = model
    self.train_count = train_count
    self.valid_count = valid_count
    
  def get_size(self, dataset: tf.data.Dataset) -> int:
    size = 0
    for _ in dataset:
      size += 1
    return size

  def run(self):
    run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    tensor_board_dir = os.path.join(os.path.abspath(os.curdir), 'logs', self.model.name, run_timestamp)
    tensor_board_callback = callbacks.TensorBoard(log_dir=tensor_board_dir)
    mongo_db = MongoDbClient()
    batch_size = 100

    train_pairs = list(mongo_db.get_pairs_collection().find({ "partition": "train", "language": "python" }).limit(self.train_count))
    positive_pairs_db = self.create_tf_dataset(train_pairs)
    train_dataset = self.generate_dataset_with_negative_samples(positive_samples=positive_pairs_db).batch(batch_size).shuffle(100)

    validation_pairs = list(mongo_db.get_pairs_collection().find({ "partition": "valid", "language": "python" }).limit(self.valid_count))
    valid_dataset = self.generate_dataset_with_negative_samples(self.create_tf_dataset(validation_pairs)).batch(batch_size)

    self.model.fit(
      train_dataset,
      validation_data=valid_dataset,
      epochs=10,
      batch_size=batch_size,
      callbacks=[tensor_board_callback]
    )
    self.model.save(os.path.join(os.path.abspath(os.curdir), f'models/{self.model.name}-{run_timestamp}'))


  def create_tf_dataset(self, pairs: List[MongoDbPairDoc]) -> tf.data.Dataset:
    def dataset_generator():
      with EmbeddingDataset('embeddings') as embedding_dataset:
        for pair in pairs:
          embedding = embedding_dataset.get(pair["id"])
          if embedding is None:
            raise ValueError(f"{pair['id']} from partition {pair['partition']} was not found")

          yield {
            "code_embedding": embedding["code_embedding"],
            "comment_embedding": embedding["comment_embedding"],
          }
    
    return tf.data.Dataset.from_generator(dataset_generator, output_types={
      "code_embedding": tf.float32, 
      "comment_embedding": tf.float32,
    })
  
  def generate_dataset_with_negative_samples(self, positive_samples: tf.data.Dataset) -> tf.data.Dataset:
    negative_samples = positive_samples.shuffle(100)
    def generator():
      for positive_sample, negative_sample in zip(positive_samples, negative_samples):
        yield {
          "code_embedding": positive_sample['code_embedding'],
          "comment_embedding": positive_sample['comment_embedding']
        }, 1.0
        yield {
          "code_embedding": positive_sample['code_embedding'],
          "comment_embedding": negative_sample['comment_embedding']
        }, 0.0

    return tf.data.Dataset.from_generator(generator, output_types=({
      "code_embedding": tf.float32,
      "comment_embedding": tf.float32,
    }, tf.float32))
