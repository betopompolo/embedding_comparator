from typing import Dict, List, Literal
from keras import Model, Sequential
from keras.layers import Dense, Input, Concatenate, Dropout
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy, Precision, Recall
from keras.optimizers import Adam

hidden_layer_activation = 'tanh'
output_activation = 'sigmoid'

def build_dense_model(num_hidden_layers: Literal[2, 4, 6], input_shape, model_name: str):
  dense_layers: Dict[int, List] = {
    2: [
      Dense(100, activation=hidden_layer_activation),
      Dense(50, activation=hidden_layer_activation),
    ],
    4: [
      Dense(200, activation=hidden_layer_activation),
      Dense(150, activation=hidden_layer_activation),
      Dense(100, activation=hidden_layer_activation),
      Dense(50, activation=hidden_layer_activation),
    ], 
    6: [
      Dense(300, activation=hidden_layer_activation),
      Dense(250, activation=hidden_layer_activation),
      Dense(200, activation=hidden_layer_activation),
      Dense(150, activation=hidden_layer_activation),
      Dense(100, activation=hidden_layer_activation),
      Dense(50, activation=hidden_layer_activation),
    ], 
  }

  code_input = Input(
    shape=input_shape,
    name="code_embedding",
  )
  comment_input = Input(
    shape=input_shape,
    name="comment_embedding",
  )

  concatenated_inputs = Concatenate()([code_input, comment_input])
  dropout = Dropout(0.2)(concatenated_inputs)
  hidden_layers = Sequential(dense_layers[num_hidden_layers], name="hidden_layers")(dropout)
  output = Dense(1, activation=output_activation, name="output")(hidden_layers)
  model = Model(
    inputs=[code_input, comment_input],
    outputs=output,
    name=model_name
  )

  model.compile(
    optimizer=Adam(),
    loss=BinaryCrossentropy(),
    metrics=[
      BinaryAccuracy(),
      Precision(name="precision"),
      Recall(name="recall"),
    ],
  )

  return model
