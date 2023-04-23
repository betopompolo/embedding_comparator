from typing import Tuple

from keras import Model, Sequential
from keras.layers import Dense
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
import tensorflow as tf

ModelInput = Tuple[tf.Tensor, tf.Tensor]

class EmbeddingComparator():
  _model: Model
  _model_dir = 'model_lite/emb_comparator' # TODO: Using lite! keep it?

  def __init__(self, activation='relu') -> None:
    self._model = Sequential([
        Dense(128, activation=activation),
        Dense(1, activation='sigmoid')
    ])

    self._model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss=BinaryCrossentropy(),
    )

  def fit(self, inputs: ModelInput, batch_size: int, epochs: int, steps_count: int):
    self._model.fit(
      inputs,
      batch_size=batch_size,
      epochs=epochs,
      steps_per_epoch=steps_count,
      shuffle=False,
    )

  def save(self):
    print(f'saving model at "{self._model_dir}"...')
    self._model.save(self._model_dir)
  
  def load(self):
    print(f'loading model from "{self._model_dir}"...')
    self._model = tf.keras.models.load_model(self._model_dir)

  def summary(self):
    self._model.summary()

  def predict(self, inputs, hide_logs = False):
    return self._model.predict(inputs, verbose=0 if hide_logs else 'auto') # type: ignore
