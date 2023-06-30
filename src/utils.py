from tensorflow_addons.metrics import F1Score
from keras import Sequential, Model
from keras.layers import Input, Dense
from keras.losses import BinaryCrossentropy
from keras.metrics import Accuracy, Precision, Recall
from keras.optimizers import Adam
from typing import Dict
from experiment_parameters import ExperimentParameters

from models import Language, Partition

def build_model(num_hidden_layers: int):
  input_shape = (encoder_seq_len * encoder_hidden_size * 2)
  hidden_layer_activation = 'relu'
  output_activation = 'sigmoid'
  dense_layers = {
    2: [
      Dense(100, activation=hidden_layer_activation),
      Dense(50, activation=hidden_layer_activation),
    ],
    4: [
      Dense(400, activation=hidden_layer_activation),
      Dense(200, activation=hidden_layer_activation),
      Dense(100, activation=hidden_layer_activation),
      Dense(50, activation=hidden_layer_activation),
    ], 
    8: [
      Dense(800, activation=hidden_layer_activation),
      Dense(600, activation=hidden_layer_activation),
      Dense(500, activation=hidden_layer_activation),
      Dense(400, activation=hidden_layer_activation),
      Dense(300, activation=hidden_layer_activation),
      Dense(200, activation=hidden_layer_activation),
      Dense(100, activation=hidden_layer_activation),
      Dense(50, activation=hidden_layer_activation),
    ], 
  }

  input = Input(
    shape=input_shape,
    name="embedding_input",
  )
  hidden_layers = Sequential(dense_layers[num_hidden_layers], name="hidden_layers")(input)
  output = Dense(1, activation=output_activation, name="output")(hidden_layers)
  model = Model(
    inputs=input,
    outputs=output,
  )
  
  model.compile(
    optimizer=Adam(),
    loss=BinaryCrossentropy(),
    metrics=[
      Accuracy(),
      Precision(),
      Recall(),
      F1Score(
        num_classes=2,
        average='macro',
        threshold=0.5,
      ),
    ]
  )

  return model

def decode_tensor_string(tensor) -> str:
  return tensor.numpy().decode('utf-8')

def get_model_path(params: ExperimentParameters) -> str:
  return f'models/{params.name}'

encoder_seq_len = 512
encoder_hidden_size = 384
