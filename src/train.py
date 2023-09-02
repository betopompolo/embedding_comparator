import os
import tensorflow as tf
from datetime import datetime
from typing import Iterator
from keras import Model, callbacks
from embedding_dataset import EmbeddingDataset
from mongo_db_client import MongoDbClient, MongoDbPairDoc
from runnable import Runnable


class Train(Runnable):
  def __init__(self, model: Model, train_count: int | None = None) -> None:
    self.model = model
    self.embedding_dataset = EmbeddingDataset('embeddings_db')
    self.train_count = train_count
    

  def run(self):
    tensor_board_dir = os.path.join(os.path.abspath(os.curdir), 'logs', self.model.name, datetime.now().strftime('%Y%m%d-%H%M%S'))
    tensor_board_callback = callbacks.TensorBoard(log_dir=tensor_board_dir)
    mongo_db = MongoDbClient()
    batch_size = 100

    train_pairs_count = self.train_count if self.train_count is not None else mongo_db.get_pairs_collection().count_documents({ "partition": "train", "language": "python" })
    train_pairs = mongo_db.get_pairs_collection().find({ "partition": "train", "language": "python" }).limit(train_pairs_count)
    positive_pairs_db = self.create_tf_dataset(train_pairs)
    train_dataset = self.generate_dataset_with_negative_samples(positive_samples=positive_pairs_db, positive_samples_len=train_pairs_count).batch(batch_size).shuffle(int(train_pairs_count * 0.2))

    validation_pairs_count = int(train_pairs_count * 0.4)
    validation_pairs = mongo_db.get_pairs_collection().find({ "partition": "valid", "language": "python" }).limit(validation_pairs_count)
    valid_dataset = self.generate_dataset_with_negative_samples(self.create_tf_dataset(validation_pairs), validation_pairs_count).batch(batch_size)

    self.model.fit(
      train_dataset,
      validation_data=valid_dataset,
      epochs=10,
      callbacks=[tensor_board_callback]
    )
    self.model.save(os.path.join(os.path.abspath(os.curdir), f'models/{self.model.name}'))


  def create_tf_dataset(self, pairs: Iterator[MongoDbPairDoc]) -> tf.data.Dataset:
    def dataset_generator():
      for pair in pairs:
        embedding = self.embedding_dataset.get(pair["id"])
        if embedding is None:
          continue

        yield {
          "code_embedding": embedding["code_embedding"],
          "comment_embedding": embedding["comment_embedding"],
        }
    
    return tf.data.Dataset.from_generator(dataset_generator, output_types={
      "code_embedding": tf.float32, 
      "comment_embedding": tf.float32,
    })
  
  def generate_dataset_with_negative_samples(self, positive_samples: tf.data.Dataset, positive_samples_len: int) -> tf.data.Dataset:
    negative_samples = positive_samples.shuffle(int(positive_samples_len * 0.2))
    def generator():
      for positive_sample, negative_sample in zip(positive_samples, negative_samples):
        yield {
          "code_embedding": positive_sample['code_embedding'],
          "comment_embedding": positive_sample['comment_embedding']
        }, 1
        yield {
          "code_embedding": positive_sample['code_embedding'],
          "comment_embedding": negative_sample['comment_embedding']
        }, 0

    return tf.data.Dataset.from_generator(generator, output_types=({
      "code_embedding": tf.float32,
      "comment_embedding": tf.float32,
    }, tf.int8))
