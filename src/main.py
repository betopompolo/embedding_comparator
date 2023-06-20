from datetime import datetime
import os
from typing import List, cast
from keras import Model, Sequential
from keras.layers import Input, Dense, Concatenate, Reshape
import tensorflow as tf
from tqdm import tqdm

from embedding_generator_default import EmbeddingGeneratorDefault
from experiment_parameters import ExperimentParameters
from models import MongoId, PairDbDoc, Result
from mongo_db_client import MongoDbClient
from pre_processer_default import PreProcesserDefault
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from result_analyzers import ResultAnalyzerBinary

from utils import encoder_hidden_size, encoder_seq_len

db_client = MongoDbClient()
embedding_generator = EmbeddingGeneratorDefault()
pre_processer = PreProcesserDefault()

def generate_code_embeddings(code_tokens: List[str]):
  return embedding_generator.from_code(pre_processer.process_code(code_tokens))

def generate_comment_embeddings(text_tokens: List[str]):
  return embedding_generator.from_text(pre_processer.process_text(text_tokens))

def get_model_path(params: ExperimentParameters) -> str:
  return f'models/{params.name}'


transformer_embedding_shape = (encoder_seq_len, encoder_hidden_size)
experiments = [
  ExperimentParameters(
    name="experiment_1",
    num_hidden_layers=2,
    programming_languages=['java', 'python'],
  ),
  ExperimentParameters(
    name="experiment_2",
    num_hidden_layers=4,
    programming_languages=['java', 'python'],
  ),
  ExperimentParameters(
    name="experiment_3",
    num_hidden_layers=8,
    programming_languages=['java', 'python'],
  ),
  ExperimentParameters(
    name="experiment_4",
    num_hidden_layers=2,
    programming_languages=['python'],
  ),
  ExperimentParameters(
    name="experiment_5",
    num_hidden_layers=4,
    programming_languages=['python'],
  ),
  ExperimentParameters(
    name="experiment_6",
    num_hidden_layers=8,
    programming_languages=['python'],
  ),
]
experiment_parameters = experiments[2]

def get_model(path: str | None = None):
  if path is not None:
    return cast(Model, tf.keras.models.load_model(path))

  hidden_layer_activation = 'relu'
  output_activation = 'sigmoid'
  code_input = Input(shape=transformer_embedding_shape, name="code_input")
  comment_input = Input(
    shape=transformer_embedding_shape, name="comment_input")

  reshaped_code = Reshape(target_shape=(-1, ), name="reshape_code")(code_input)
  reshaped_comment = Reshape(target_shape=(-1, ), name="reshape_comment")(comment_input)

  merged = Concatenate(name="joint_embedding")(
    [reshaped_code, reshaped_comment])
  
  dense_layers = {
    2: [
      Dense(400, activation=hidden_layer_activation),
      Dense(200, activation=hidden_layer_activation),
    ],
    4: [
      Dense(600, activation=hidden_layer_activation),
      Dense(400, activation=hidden_layer_activation),
      Dense(200, activation=hidden_layer_activation),
      Dense(100, activation=hidden_layer_activation),
    ], 
    8: [
      Dense(800, activation=hidden_layer_activation),
      Dense(600, activation=hidden_layer_activation),
      Dense(500, activation=hidden_layer_activation),
      Dense(400, activation=hidden_layer_activation),
      Dense(300, activation=hidden_layer_activation),
      Dense(250, activation=hidden_layer_activation),
      Dense(200, activation=hidden_layer_activation),
      Dense(100, activation=hidden_layer_activation),
    ], 
  }

  hidden_layers = Sequential(dense_layers[experiment_parameters.num_hidden_layers], name="hidden_layers")

  hidden = hidden_layers(merged)
  output = Dense(1, activation=output_activation, name="output")(hidden)

  model = Model(inputs=[code_input, comment_input], outputs=output)
  model.compile(
    optimizer=Adam(),
    loss=BinaryCrossentropy(),
    metrics=['accuracy']
  )

  return model

mode = os.getenv('MODE')
mode = mode.lower() if mode != None else ''

print(f'running {experiment_parameters.name}...')

if mode == 'valid':
  results_analyzer = ResultAnalyzerBinary()
  results = []
  model_path = get_model_path(experiment_parameters)
  validation_model = get_model(path=model_path)

  for query_doc in tqdm(db_client.get_queries_collection().find({ "language": { "$in": experiment_parameters.programming_languages } }), total=db_client.get_queries_collection().count_documents({}), desc="Validating CSNet queries"):
      pair_doc = query_doc['pair_doc']
      code_emb, comment_emb = generate_code_embeddings(
          pair_doc['code_tokens']), generate_comment_embeddings(query_doc['query'].split())
      prediction = validation_model.predict(
          x=[code_emb, comment_emb], verbose="0")
      similarity = prediction.item()

      results.append(Result(
          code_url=query_doc['url'],
          relevance=query_doc['relevance'],
          similarity=similarity,
      ))

  def write_results(result_list: List[Result]):
    with open(f'results/{datetime.now().isoformat()}.csv', 'w') as file:
      file.write(f'model: {model_path}\n')
      for result in tqdm(result_list, desc='Writing results'):
        file.write(f'{result.similarity}, {result.relevance}\n')

  write_results(results)
  results_analyzer.print_results(results)
elif mode == 'summary':
  get_model().summary()
else:
  training_model = get_model()
  pairs = db_client.get_pairs_collection()
  train_samples = 5000
  num_epochs = 10
  batch_size = 1
  model_path = get_model_path(experiment_parameters)

  def random_pair(exclude_id: MongoId) -> PairDbDoc:
      return pairs.aggregate([
          {'$sample': {'size': 1}},
          {'$match': {'_id': {'$ne': exclude_id}}}
      ]).next()

  def training_dataset():
    pairs_query = {
      "language": {
        "$in": experiment_parameters.programming_languages
      }
    }
    for pair_doc in pairs.find(pairs_query).limit(int(train_samples / 2)):
      code_emb, comment_emb, target = generate_code_embeddings(pair_doc['code_tokens']), generate_comment_embeddings(pair_doc['comment_tokens']), tf.ones(shape=(batch_size, ))
      yield ((code_emb, comment_emb), target)

      negative_doc = random_pair(exclude_id=pair_doc['_id'])
      negative_code_emb, negative_comment_emb, negative_target = generate_code_embeddings(pair_doc['code_tokens']), generate_comment_embeddings(negative_doc['comment_tokens']), tf.zeros(shape=(batch_size, ))
      yield ((negative_code_emb, negative_comment_emb), negative_target)
  
  embedding_spec = tf.TensorSpec(shape=(batch_size, encoder_seq_len, encoder_hidden_size), dtype=tf.float64) # type: ignore
  target_spec = tf.TensorSpec(shape=(batch_size, ), dtype=tf.int32) # type: ignore
  ds = tf.data.Dataset.from_generator(
    training_dataset,
    output_signature=((embedding_spec, embedding_spec), target_spec),
  ).shuffle(buffer_size=int(train_samples * 0.2))
  training_model.fit(ds, epochs=num_epochs, steps_per_epoch=train_samples/num_epochs)
  training_model.save(model_path)
