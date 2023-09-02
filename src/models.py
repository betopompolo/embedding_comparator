from typing import Dict, List, Literal
import tensorflow as tf
from keras import Model, Sequential
from keras.layers import Dense, Input, Concatenate
from keras.losses import Loss, BinaryCrossentropy
from keras.utils import losses_utils
from keras.metrics import BinaryAccuracy, Precision, Recall
from keras.optimizers import Adam

input_shape = (480)
hidden_layer_activation = 'tanh'
output_activation = 'sigmoid'

def build_dense_model(num_hidden_layers: Literal[2, 4, 8]):
  dense_layers: Dict[int, List] = {
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

  code_input = Input(
    shape=input_shape,
    name="code_embedding",
  )
  comment_input = Input(
    shape=input_shape,
    name="comment_embedding",
  )

  concatenated_inputs = Concatenate()([code_input, comment_input])
  hidden_layers = Sequential(dense_layers[num_hidden_layers], name="hidden_layers")(concatenated_inputs)
  output = Dense(1, activation=output_activation, name="output")(hidden_layers)
  model = Model(
    inputs=[code_input, comment_input],
    outputs=output,
    name="embedding_comparator_dense"
  )

  model.compile(
    optimizer=Adam(),
    loss=BinaryCrossentropy(),
    metrics=[
      BinaryAccuracy(),
      Precision(name="precision"),
      Recall(name="recall"),
      # f1_score, # TODO: Reactivate
    ],
  )

  return model


# class ConstrastiveLoss(Loss):
#   def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name="constrastive_loss", margin=1):
#     self.margin = margin
#     super().__init__(reduction, name)

#   def call(self, y_true, y_pred):
#     print(f'{type(y_true)} {type(y_pred)}')
#     square_pred = tf.math.square(y_pred)
#     margin_square = tf.math.square(tf.math.maximum(self.margin - (y_pred), 0))
#     return tf.math.reduce_mean(
#       (1 - y_true) * square_pred + (y_true) * margin_square
#     )