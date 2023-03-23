from typing import Tuple

from keras import Model, Sequential
from keras.layers import Dense, Input
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
import tensorflow as tf
import tensorflow_addons as tfa


ModelInput = Tuple[tf.Tensor, tf.Tensor]


class EmbeddingComparator():
    _model: Model
    _model_weights_path = 'model/emb_comparator'

    def __init__(self, activation='relu') -> None:
        self._model = Sequential([
            Dense(128, activation=activation),
            Dense(1, activation='sigmoid')
        ])

        self._model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss=BinaryCrossentropy(),
        )

    def fit(self, inputs: ModelInput, batch_size=64, epochs=5):
        self._model.fit(
            inputs,
            batch_size=batch_size,
            epochs=epochs,
        )
    def save(self):
        self._model.save_weights(self._model_weights_path)
    
    def load(self):
        self._model.load_weights(self._model_weights_path)

    def predict(self, inputs):
        return self._model.predict(inputs)
