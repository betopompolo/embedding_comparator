from keras import Model, Sequential
from keras.layers import Dense
from keras.layers.core import Dropout
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
import tensorflow as tf

from models import EmbeddingComparator, ModelInput

class EmbeddingComparatorDense(EmbeddingComparator):
  __model: Model
  __model_dir = 'models/'

  def __init__(self) -> None:
    self.__model = Sequential([
      Dense(400, activation='tanh'),
      Dropout(rate=0.1),
      Dense(1, activation='sigmoid')
    ])

    self.__model.compile(
      optimizer=Adam(learning_rate=0.01),
      loss=BinaryCrossentropy(),
    )

  def fit(self, inputs: ModelInput, batch_size: int, epochs: int):
    self.__model.fit(
      inputs,
      batch_size=batch_size,
      epochs=epochs,
      shuffle=False,
    )

  def save(self, file_name: str):
    model_path = self.__model_dir + file_name
    print(f'saving model at "{model_path}"...')
    self.__model.save(model_path)
  
  def load(self, file_name: str):
    model_path = self.__model_dir + file_name
    print(f'loading model from "{model_path}"...')
    self.__model = tf.keras.models.load_model(model_path)

  def predict(self, inputs):
    return self.__model.predict(inputs, verbose=0)[0][0] # type: ignore

  def summary(self):
    self.__model.summary()