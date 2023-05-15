from keras import Model, Sequential
from keras.layers import Dropout, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
import tensorflow as tf

from models import EmbeddingComparator, ModelInput

class EmbeddingComparatorDevise(EmbeddingComparator):
  __model: Model
  __model_dir = 'models/'

  def fit(self, inputs: ModelInput, batch_size: int, epochs: int):
    if hasattr(self, '__model') == False:
      # TODO: Based on DeVise https://github.com/fg91/DeViSE-zero-shot-classification/blob/master/DeViSE%20-%20A%20Deep%20Visual-Semantic%20Embedding%20Model.ipynb
      activation = 'relu'
      self.__model = Sequential([
        BatchNormalization(axis=1),
        Dense(units=512, activation=activation),
        Dropout(0.01),
        Dense(units=300, activation=activation),
        Dense(1, activation='sigmoid')
      ])

      self.__model.compile(
        optimizer=Adam(beta_1=0.9, beta_2=0.09),
        loss=BinaryCrossentropy(),
      )
    
    self.__model.fit(
      inputs,
      batch_size=batch_size,
      epochs=epochs,
      steps_per_epoch=1000, # TODO: get this value dynamically
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
    return self.__model.predict(inputs)[0][0]

  def summary(self):
    self.__model.summary()