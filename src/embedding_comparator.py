from typing import Tuple

from keras import Model, Sequential
from keras.layers import Dense
from keras.layers.core import Dropout
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
import tensorflow as tf

ModelInput = Tuple[tf.Tensor, tf.Tensor]

class EmbeddingComparator():
  _model: Model

  def __init__(self) -> None:
    self._model = Sequential([
      Dense(400, activation='tanh'),
      Dropout(rate=0.1),
      Dense(1, activation='sigmoid')
    ])

    self._model.compile(
      optimizer=Adam(learning_rate=0.01),
      loss=BinaryCrossentropy(),
    )

  def fit(self, inputs: ModelInput, batch_size: int, epochs: int):
    self._model.fit(
      inputs,
      batch_size=batch_size,
      epochs=epochs,
      shuffle=False,
    )

  def save(self, model_dir: str):
    print(f'saving model at "{model_dir}"...')
    self._model.save(model_dir)
  
  def load(self, model_dir: str):
    print(f'loading model from "{model_dir}"...')
    return tf.keras.models.load_model(model_dir)

  def predict(self, inputs, hide_logs = False):
    return self._model.predict(inputs, verbose=0 if hide_logs else 'auto') # type: ignore
