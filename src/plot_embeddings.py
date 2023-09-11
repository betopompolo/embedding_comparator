import os
from typing import List
from matplotlib import pyplot as plt
import more_itertools
import tensorflow as tf
from embedding_dataset import EmbeddingDataset
from mongo_db_client import MongoDbClient, MongoDbPairDoc
from runnable import Runnable


class PlotEmbeddings(Runnable):
  def __init__(self, plots_dir = '') -> None:
    self.plots_dir = plots_dir if len(plots_dir) > 0 else os.path.join(os.path.abspath(os.curdir), 'plots')

    if not os.path.isdir(self.plots_dir):
      os.mkdir(self.plots_dir)

  def run(self):
    mongo_db = MongoDbClient()

    pairs = list(mongo_db.get_pairs_collection().find({ "partition": "train", "language": "python" }).limit(50))
    positive_pairs_db = self.create_tf_dataset(pairs)
    train_dataset = self.generate_dataset_with_negative_samples(positive_pairs_db)
    sample = 0
    for [positive, pos_target], [negative, neg_target] in more_itertools.chunked(train_dataset.as_numpy_iterator(), 2):
      self.save_plot_emb_pairs(positive, negative, f'sample_{sample}')
      sample += 1
    
  
  def save_plot_emb_pairs(self, positive, negative, title: str):
    fig, axes = plt.subplots(2, 2, constrained_layout=True)

    code_graph, comment_graph, neg_code_graph, neg_comment_graph = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

    code_graph.plot(positive['code_embedding'], color='red')
    code_graph.set_title('Code')
    comment_graph.plot(positive['comment_embedding'])
    comment_graph.set_title('Comment')

    neg_code_graph.plot(negative['code_embedding'], color='red')
    neg_code_graph.set_title('Code')
    neg_comment_graph.plot(negative['comment_embedding'])
    neg_comment_graph.set_title('Negative comment')

    fig.suptitle(title)
    
    fig_path = os.path.join(self.plots_dir, title)
    fig.savefig(fig_path)

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