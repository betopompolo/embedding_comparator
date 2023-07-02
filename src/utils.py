from keras import Model, Sequential, backend as K
from keras.layers import Dense, Input
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy, Precision, Recall
from keras.optimizers import Adam

from experiment_parameters import ExperimentParameters

def f1_score(y_true, y_pred):
  def calc_recall():
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

  def calc_precision():
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

  precision = calc_precision()
  recall = calc_recall()

  return 2*((precision*recall)/(precision+recall+K.epsilon()))

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

  threshold = 0.5
  
  model.compile(
    optimizer=Adam(),
    loss=BinaryCrossentropy(),
    metrics=[
      BinaryAccuracy(threshold=threshold),
      Precision(thresholds=threshold),
      Recall(thresholds=threshold),
      f1_score,
    ],
  )

  return model

def decode_tensor_string(tensor) -> str:
  return tensor.numpy().decode('utf-8')

def get_model_path(params: ExperimentParameters) -> str:
  return f'models/{params.name}'

encoder_seq_len = 512
encoder_hidden_size = 384
