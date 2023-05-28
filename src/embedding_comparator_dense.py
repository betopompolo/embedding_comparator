from keras import Model, Sequential
from keras.layers import Dense, Flatten, Input
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
import tensorflow as tf

from models import EmbeddingComparator, ModelInput

from utils import encoder_hidden_size, encoder_seq_len

class EmbeddingComparatorDense(EmbeddingComparator):
  __model: Model
  name: str

  def __init__(self, name: str, batch_size: int) -> None:
    self.name = name
    self.batch_size = batch_size

    self.__model = Sequential([
      Input(shape=(encoder_seq_len * encoder_hidden_size * 2), batch_size=batch_size),
      Dense(400, activation='tanh'),
      Dense(300, activation='tanh'),
      Dense(200, activation='tanh'),
      Dense(100, activation='tanh'),
      Dense(1, activation='relu')
    ])

    self.__model.compile(
      optimizer=Adam(learning_rate=0.01),
      loss=BinaryCrossentropy(),
    )

  def fit(self, inputs: ModelInput, epochs: int, batch_count: int):
    self.__model.fit(
      inputs,
      epochs=epochs,
      steps_per_epoch=batch_count,
      use_multiprocessing=True
    )

  def save(self):
    model_path = 'models/' + self.name
    print(f'saving model at "{model_path}"...')
    self.__model.save(model_path)
  
  def load(self):
    model_path = 'models/' + self.name
    print(f'loading model from "{model_path}"...')
    self.__model = tf.keras.models.load_model(model_path)

  def predict(self, inputs):
    prediction = self.__model.predict(inputs, verbose=0) # type: ignore
    return prediction[0][0]

  def summary(self):
    self.__model.summary()