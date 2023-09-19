import os
from typing import List
from matplotlib import pyplot as plt
import more_itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from embedding_dataset import EmbeddingDataset
from embedding_generator import EmbeddingPair
from mongo_db_client import MongoDbClient, MongoDbPairDoc
from runnable import Runnable


class DataAnalysis(Runnable):
  def __init__(self, analysis_dir = '') -> None:
    self.analysis_dir = analysis_dir if len(analysis_dir) > 0 else os.path.join(os.path.abspath(os.curdir), 'analysis')

    if not os.path.isdir(self.analysis_dir):
      os.mkdir(self.analysis_dir)

  def run(self):
    mongo_db = MongoDbClient()

    pairs = list(mongo_db.get_pairs_collection().find({ "partition": "train", "language": "python" }).limit(50))
    positive_pairs_db = self.create_tf_dataset(pairs)
    train_dataset = self.generate_dataset_with_negative_samples(positive_pairs_db)

    self.embeddings_analysis(train_dataset)
    self.raw_data_analysis(train_dataset)
    
  
  def raw_data_analysis(self, train_dataset: tf.data.Dataset):
    ids = [pair['id'].decode('utf-8') for pair, target in train_dataset.as_numpy_iterator()]
    pairs: dict[str, MongoDbPairDoc] = { pair['id']: pair for pair in MongoDbClient().get_pairs_collection().find({ "id": { "$in": ids } }) } 
    raw_data_path = os.path.join(self.analysis_dir, 'raw_data.csv')

    df_data = []
    sample = 0

    for [positive_emb_pair, pos_target], [negative_emb_pair, neg_target] in more_itertools.chunked(train_dataset.as_numpy_iterator(), 2):
      positive_pair = pairs[positive_emb_pair['id'].decode('utf-8')]
      negative_pair = pairs[negative_emb_pair['id'].decode('utf-8')]

      if positive_pair is None:
        raise ValueError(f"positive pair with id {positive_emb_pair['id']} was not found")
      
      if negative_pair is None:
        raise ValueError(f"negative pair with id {negative_emb_pair['id']} was not found")

      df_data.append({
        "sample": sample,
        "code": positive_pair['code_tokens'],
        "positive_comment": positive_pair['comment_tokens'],
        "negative_comment": negative_pair['comment_tokens'],
      })
      sample += 1

    df = pd.DataFrame(df_data)
    df.to_csv(raw_data_path)

      
    
  def embeddings_analysis(self, train_dataset):
    sample = 0
    for [positive, pos_target], [negative, neg_target] in more_itertools.chunked(train_dataset.as_numpy_iterator(), 2):
      self.save_plot_emb_pairs(positive, negative, f'sample-{sample}')
      sample += 1

  def save_raw_data(self, positive_pair: EmbeddingPair, negative_pair: EmbeddingPair, title: str):
    path = os.path.join(self.analysis_dir, f'{title}.csv')
    with open(path, 'w') as file:
      file.write(f"{positive_pair['code_embedding']},{positive_pair['comment_embedding']},{negative_pair['comment_embedding']}\n")

  def save_plot_emb_pairs(self, positive, negative, title: str):
    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    code_graph, comment_graph, neg_comment_graph, diff_graph = axes[0][0], axes[0][1], axes[1][0], axes[1][1]
    
    comment_diff = np.absolute(positive['comment_embedding'] - negative['comment_embedding'])

    code_graph.plot(positive['code_embedding'], color='#14d0c5')
    code_graph.set_title('Code')

    comment_graph.plot(positive['comment_embedding'], color='#b1d697')
    comment_graph.set_title('Comment')

    neg_comment_graph.plot(negative['comment_embedding'], color='#6db474')
    neg_comment_graph.set_title('Negative comment')

    diff_graph.plot(comment_diff, color='#8db670')
    diff_graph.set_title('Comment Diff')

    fig.suptitle(title)
    
    fig_path = os.path.join(self.analysis_dir, title)
    fig.savefig(fig_path)
    plt.close(fig)
    

  def create_tf_dataset(self, pairs: List[MongoDbPairDoc]) -> tf.data.Dataset:
    def dataset_generator():
      with EmbeddingDataset('embeddings') as embedding_dataset:
        for pair in pairs:
          embedding = embedding_dataset.get(pair["id"])
          if embedding is None:
            raise ValueError(f"{pair['id']} from partition {pair['partition']} was not found")

          yield {
            "id": embedding['id'],
            "code_embedding": embedding["code_embedding"],
            "comment_embedding": embedding["comment_embedding"],
          }
    
    return tf.data.Dataset.from_generator(dataset_generator, output_types={
      "id": tf.string,
      "code_embedding": tf.float32, 
      "comment_embedding": tf.float32,
    })
  
  
  def generate_dataset_with_negative_samples(self, positive_samples: tf.data.Dataset) -> tf.data.Dataset:
    negative_samples = positive_samples.shuffle(100)
    def generator():
      for positive_sample, negative_sample in zip(positive_samples, negative_samples):
        yield {
          "id": positive_sample['id'],
          "code_embedding": positive_sample['code_embedding'],
          "comment_embedding": positive_sample['comment_embedding']
        }, 1.0
        yield {
          "id": negative_sample['id'],
          "code_embedding": positive_sample['code_embedding'],
          "comment_embedding": negative_sample['comment_embedding']
        }, 0.0

    return tf.data.Dataset.from_generator(generator, output_types=({
      "id": tf.string,
      "code_embedding": tf.float32,
      "comment_embedding": tf.float32,
    }, tf.float32))