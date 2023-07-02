from datetime import datetime
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Literal, TypedDict
from sklearn.model_selection import StratifiedKFold
from keras import callbacks
from tqdm import tqdm
from embedding_concat_default import EmbeddingConcatDefault
from embedding_generator_default import EmbeddingGeneratorDefault

from experiment_parameters import ExperimentParameters
from models import EmbeddingGenerator, Partition, Runnable

from mongo_db_client import MongoDbClient
from utils import build_model

class CrossValidationSample(TypedDict):
  code_tokens: List[str]
  comment_tokens: List[str]
  target: Literal[0] | Literal[1]

@dataclass
class CrossValidation(Runnable):
  db_client = MongoDbClient()
  experiments: List[ExperimentParameters]
  """
  validation_samples=1000; epoch=5 -> 538s
  """
  validation_samples = 1000
  embedding_concat = EmbeddingConcatDefault()

  def run(self):
    for experiment in self.experiments:
      kfold = StratifiedKFold(
        n_splits=10, 
        shuffle=True, 
        random_state=42,
      )
      model = build_model(experiment.num_hidden_layers)
      embedding_generator = EmbeddingGeneratorDefault()
      logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
      tensor_board_callback = callbacks.TensorBoard(log_dir=logdir)

      (inputs, targets) = self.generate_model_input(
        self.get_samples('train') + self.get_samples('test'),
        embedding_generator,
      )

      for train, test in kfold.split(inputs, targets):
        model.fit(
          inputs[train],
          targets[train],
          epochs=10,
          callbacks=[tensor_board_callback]
        )
        model.evaluate(inputs[test], targets[test], verbose=0) # type: ignore

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

    for sample in tqdm(samples, desc="Generating inputs and targets"):
      concatenated = self.embedding_concat.concatenate(
        embedding_generator.from_code(sample['code_tokens']), 
        embedding_generator.from_text(sample['comment_tokens']),
        reshape=(-1,),
      )

      inputs.append(concatenated)
      targets.append(sample['target'])

    return (np.array(inputs), np.array(targets))