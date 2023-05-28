from datetime import datetime
from typing import List, cast
from keras import Model
from keras.layers import Input, Dense, Concatenate
import tensorflow as tf
from tqdm import tqdm

from embedding_generator_default import EmbeddingGeneratorDefault
from models import MongoId, PairDbDoc, Result
from mongo_db_client import MongoDbClient
from pre_processer_default import PreProcesserDefault
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from result_analyzers import ResultAnalyzerSigmoid

from utils import encoder_hidden_size, encoder_seq_len

db_client = MongoDbClient()
embedding_generator = EmbeddingGeneratorDefault()
pre_processer = PreProcesserDefault()

def generate_code_embeddings(code_tokens: List[str]):
  return embedding_generator.from_code(pre_processer.process_code(code_tokens))

def generate_comment_embeddings(text_tokens: List[str]):
  return embedding_generator.from_text(pre_processer.process_text(text_tokens))

hidden_layer_activation = 'tanh'
output_activation = 'sigmoid'
num_epochs = 10
transformer_embedding_shape = (encoder_seq_len, encoder_hidden_size)
train_samples = 10
model_path = f'models/concat_{train_samples}'

def get_model(path: str | None = None):
  if path is not None:
    return cast(Model, tf.keras.models.load_model(path)) 
  
  code_input = Input(shape=transformer_embedding_shape, name="code_input")
  comment_input = Input(shape=transformer_embedding_shape, name="comment_input")

  hidden1 = Dense(64, activation=hidden_layer_activation)(code_input)
  hidden2 = Dense(32, activation=hidden_layer_activation)(comment_input)
  merged = Concatenate(name="joint_embedding")([hidden1, hidden2])
  hidden3 = Dense(16, activation=hidden_layer_activation)(merged)
  output = Dense(1, activation=output_activation)(hidden3)

  model = Model(inputs=[code_input, comment_input], outputs=output)
  model.compile(
    optimizer=Adam(),
    loss=BinaryCrossentropy(),
    metrics=['accuracy']
  )

  return model

"""Training"""
# training_model = get_model()
# labels_shape = (1, encoder_seq_len, 1)
# pairs = db_client.get_pairs_collection()

# def random_pair(exclude_id: MongoId) -> PairDbDoc:
#   return pairs.aggregate([
#     { '$sample': { 'size': 1 } },
#     { '$match': { '_id': { '$ne': exclude_id } } }
#   ]).next()

# for pair_doc in pairs.find({}).limit(train_samples):
#   code_emb, comment_emb = generate_code_embeddings(pair_doc['code_tokens']), generate_comment_embeddings(pair_doc['comment_tokens'])
#   training_model.fit([code_emb, comment_emb], tf.ones(shape=labels_shape), epochs=num_epochs, shuffle=False)

#   negative_doc = random_pair(exclude_id=pair_doc['_id'])
#   negative_code_emb, negative_comment_emb = generate_code_embeddings(negative_doc['code_tokens']), generate_comment_embeddings(negative_doc['comment_tokens'])
#   training_model.fit([negative_code_emb, negative_comment_emb], tf.zeros(shape=labels_shape), epochs=num_epochs, shuffle=False)

# training_model.save(model_path)


"""Validation"""
results_analyzer = ResultAnalyzerSigmoid()
results = []
validation_model = get_model(path=model_path)

for query_doc in tqdm(db_client.get_queries_collection().find({}), total=db_client.get_queries_collection().count_documents({}), desc="Validating CSNet queries"):
  pair_doc = query_doc['pair_doc']
  code_emb, comment_emb = generate_code_embeddings(pair_doc['code_tokens']), generate_comment_embeddings(query_doc['query'].split())
  similarity = validation_model.predict(x=[code_emb, comment_emb], verbose="0")[0][0][0]

  results.append(Result(
    code_url=query_doc['url'],
    relevance=query_doc['relevance'],
    similarity=similarity,
  )) 

def write_results(result_list: List[Result], file_name_suffix: str = "result"):
  with open(f'results/{datetime.now().isoformat()}-{file_name_suffix}.csv', 'w') as file:
    file.write(f'Using {model_path} model\n')
    for result in tqdm(result_list, desc='Writing results'):
      file.write(f'{result.similarity}, {result.relevance}\n')

write_results(results)
results_analyzer.print_results(results)
