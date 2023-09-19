import os
import tensorflow as tf
from datetime import datetime
from typing import List
from keras import Model, callbacks
from embedding_dataset import EmbeddingDataset
from runnable import Runnable


class Train(Runnable):
  def __init__(self, model: Model, train_count: int, valid_count: int, embeddings_dataset_name: str) -> None:
    self.model = model
    self.train_count = train_count
    self.valid_count = valid_count
    self.embeddings_dataset_name = embeddings_dataset_name


  def run(self):
    run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    tensor_board_dir = os.path.join(os.path.abspath(os.curdir), 'logs', self.model.name, run_timestamp)
    tensor_board_callback = callbacks.TensorBoard(log_dir=tensor_board_dir)
    batch_size = 100

    positive_embeddings = self.create_tf_dataset()
    dataset = self.generate_dataset_with_negative_samples(positive_embeddings).shuffle(buffer_size=100).batch(batch_size)

    self.model.fit(
      dataset,
      epochs=10,
      batch_size=batch_size,
      callbacks=[tensor_board_callback]
    )
    self.model.save(os.path.join(os.path.abspath(os.curdir), f'models/{self.model.name}'))


  def create_tf_dataset(self) -> tf.data.Dataset:
    def dataset_generator():
      with EmbeddingDataset(self.embeddings_dataset_name) as embedding_dataset:
        for embedding_pair in embedding_dataset.list():
          yield {
            "code_embedding": embedding_pair["code_embedding"],
            "comment_embedding": embedding_pair["comment_embedding"],
          }
    
    return tf.data.Dataset.from_generator(dataset_generator, output_types={
      "code_embedding": tf.float32, 
      "comment_embedding": tf.float32,
    })
  
  
  def generate_dataset_with_negative_samples(self, positive_samples: tf.data.Dataset) -> tf.data.Dataset:
    negative_samples = positive_samples.shuffle(buffer_size=100)
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
