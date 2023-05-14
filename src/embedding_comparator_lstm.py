from keras import Model, Sequential
from keras.layers import Dense, Input
from keras.layers.core import Dropout
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
import tensorflow as tf

from models import EmbeddingComparator, ModelInput

class EmbeddingComparatorLSTM(EmbeddingComparator):
  _model: Model

  def fit(self, inputs: ModelInput, batch_size: int, epochs: int):
    if self._model is None:
      self._model = Sequential([
        Input(shape=(128, 786432)),
        Dense(400, activation='tanh'),
        Dropout(rate=0.1),
        Dense(1, activation='sigmoid')
      ])

      self._model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss=BinaryCrossentropy(),
      )
    
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
    self._model = tf.keras.models.load_model(model_dir)

  def predict(self, inputs):
    return self._model.predict(inputs)[0][0]

  def summary(self):
    self._model.summary()