from typing import Dict, List, Literal
import tensorflow as tf
from keras import Model, Sequential
from keras.layers import Dense, Input, Concatenate, BatchNormalization, Dot, Add, LayerNormalization, Dropout, Conv1D, AveragePooling1D, Flatten
from keras.losses import BinaryCrossentropy, Loss, losses_utils, CosineSimilarity
from keras.metrics import BinaryAccuracy, Precision, Recall
from keras.optimizers import Adam, RMSprop

input_shape = (480)
hidden_layer_activation = 'tanh'
output_activation = 'sigmoid'

def build_dense_model(num_hidden_layers: Literal[2, 4, 8], model_name: str):
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
    name=model_name
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

class ConstrastiveLoss(Loss):
  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name="constrastive_loss", margin=1):
    self.margin = margin
    super().__init__(reduction, name)

  def call(self, y_true, y_pred):
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(self.margin - (y_pred), 0))
    return tf.math.reduce_mean(
      (1 - y_true) * square_pred + (y_true) * margin_square
    )

  
def euclidean_distance(vects):
  x, y = vects
  sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
  return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


def build_siamese_model(model_name: str):
  code_input = Input(
    shape=input_shape,
    name="code_embedding",
  )
  comment_input = Input(
    shape=input_shape,
    name="comment_embedding",
  )

  # distance_layer = Lambda(euclidean_distance)([code_input, comment_input])
  distance_layer = Dot(axes=1, normalize=True)([code_input, comment_input]) # Cosine similarity
  dense_layers = Sequential([
    Dense(100, activation=hidden_layer_activation),
    Dense(50, activation=hidden_layer_activation),
  ])(distance_layer)
  normal_layer = BatchNormalization()(dense_layers)
  output_layer = Dense(1, activation="sigmoid")(normal_layer)
  siamese = Model(inputs=[code_input, comment_input], outputs=output_layer, name=model_name)
  siamese.compile(
    optimizer=RMSprop(),
    loss=ConstrastiveLoss(),
    metrics=[
      BinaryAccuracy(),
      Precision(name="precision"),
      Recall(name="recall"),
      # f1_score, # TODO: Reactivate
    ],
  )
  return siamese


def dual_encoder_model(name: str, num_projection_layers=1, projection_dims=256, dropout_rate=0.1):
  def project_embeddings(embeddings):
    projected_embeddings = Dense(units=projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = tf.nn.relu(projected_embeddings)
        x = Dense(projection_dims)(x)
        x = Dropout(dropout_rate)(x)
        x = Add()([projected_embeddings, x])
        projected_embeddings = LayerNormalization()(x)
    return projected_embeddings
  
  code_input = Input(
    shape=input_shape,
    name="code_embedding",
  )
  comment_input = Input(
    shape=input_shape,
    name="comment_embedding",
  )
  code_input_projected = project_embeddings(code_input)
  comment_input_projected = project_embeddings(comment_input)
  distance_layer = Dot(axes=1, normalize=True)([code_input_projected, comment_input_projected])

  output_layer = Dense(1, activation="sigmoid")(distance_layer)

  model = Model([code_input, comment_input], output_layer, name=name)
  model.compile(
    optimizer=Adam(),
    loss=CosineSimilarity(),
    metrics=[
      BinaryAccuracy(),
      Precision(name="precision"),
      Recall(name="recall"),
      # f1_score, # TODO: Reactivate
    ],
  )

  return model

def multilayer_raw(model_name: str, code_embedding_shape: tuple, comment_embedding_shape: tuple, num_hidden_layers: Literal[2, 4, 8]):
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

  def pre_process_layer(input_layer):
    return Sequential([
      input_layer,
      Conv1D(4, 128, activation=hidden_layer_activation),
      AveragePooling1D(pool_size=2),
      Conv1D(16, 128, activation=hidden_layer_activation),
      AveragePooling1D(pool_size=2),
      Flatten(),
      BatchNormalization(),
      Dense(10, activation=hidden_layer_activation)
    ])

  code_input = Input(
    shape=code_embedding_shape,
    name="code_embedding",
  )
  comment_input = Input(
    shape=comment_embedding_shape,
    name="comment_embedding",
  )

  code_pre_process = pre_process_layer(code_input)
  comment_pre_process = pre_process_layer(comment_input)

  concatenated_inputs = Concatenate()([code_pre_process, comment_pre_process])
  hidden_layers = Sequential(dense_layers[num_hidden_layers], name="hidden_layers")(concatenated_inputs)
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
      # f1_score, # TODO: Reactivate
    ],
  )

  return model
