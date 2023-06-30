import random
import numpy as np
from dataclasses import dataclass
from typing import List, Literal, TypedDict
from sklearn.model_selection import KFold
import tensorflow as tf
from tqdm import tqdm
from embedding_concat_default import EmbeddingConcatDefault
from embedding_generator_default import EmbeddingGeneratorDefault

from experiment_parameters import ExperimentParameters
from models import EmbeddingGenerator, Partition, Runnable

from mongo_db_client import MongoDbClient
from utils import build_model, encoder_hidden_size, encoder_seq_len

class CrossValidationSample(TypedDict):
  code_tokens: List[str]
  comment_tokens: List[str]
  target: Literal[0] | Literal[1]

@dataclass
class CrossValidation(Runnable):
  db_client = MongoDbClient()
  experiments: List[ExperimentParameters]
  validation_samples = 10
  embedding_concat = EmbeddingConcatDefault()

  def run(self):
    batch_size = 1
    for experiment in self.experiments:
      kfold = KFold(n_splits=10, shuffle=False)
      model = build_model(experiment.num_hidden_layers)
      embedding_generator = EmbeddingGeneratorDefault()

      (inputs, targets) = self.generate_model_input(
        self.get_samples('train') + self.get_samples('test'), 
        embedding_generator,
      )

      for train, test in kfold.split(inputs, targets):
        for i in train:
          model.fit(
            inputs[i],
            targets[i],
            epochs=10,
            batch_size=batch_size
          )
        # scores = model.evaluate(inputs[test], targets[test], verbose="0")
        # print(f'Score: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

  def get_samples(self, partition: Partition) -> List[CrossValidationSample]:
    pairs = list(
      self.db_client.get_pairs_collection().find({ 'partition': partition }).limit(int(self.validation_samples / 2))
    )

    negative_pairs = pairs.copy()
    random.shuffle(negative_pairs)

    samples: List[CrossValidationSample] = []
    for pair, negative_pair in zip(pairs, negative_pairs):
      samples.extend([
        {
          'code_tokens': pair['code_tokens'],
          'comment_tokens': pair['comment_tokens'],
          'target': 1,
        },
        {
          'code_tokens': pair['code_tokens'],
          'comment_tokens': negative_pair['comment_tokens'],
          'target': 0,
        },
      ])

    return samples
  
  def generate_model_input(self, samples: List[CrossValidationSample], embedding_generator: EmbeddingGenerator):
    inputs = []
    targets = []
    batch_size = 1

    for sample in tqdm(samples, desc="Generating inputs and targets"):
      concatenated = self.embedding_concat.concatenate(
        embedding_generator.from_code(sample['code_tokens']), 
        embedding_generator.from_text(sample['comment_tokens']),
        reshape=(batch_size, -1),
      )

      inputs.append(concatenated)
      targets.append(np.full((batch_size, ), sample['target']))

    return (np.array(inputs), np.array(targets))