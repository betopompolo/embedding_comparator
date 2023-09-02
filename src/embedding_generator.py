from typing import Iterator, List, TypedDict
import more_itertools
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel, AutoConfig

from mongo_db_client import MongoDbClient, MongoDbPairDoc

class EmbeddingPairBatch(TypedDict):
  pairs_ids: List[str]
  code_embeddings: np.ndarray
  comment_embeddings: np.ndarray


class EmbeddingGenerator:
  def __init__(self, code_embedding_model="microsoft/codebert-base", comment_embedding_model="bert-large-uncased") -> None:
    self.embedding_max_length = 256
    self.hidden_size = 480
    
    self.code_embedding_model = TFAutoModel.from_config(AutoConfig.from_pretrained(
      code_embedding_model,
      hidden_size=self.hidden_size,
    ))
    self.code_embedding_tokenizer = AutoTokenizer.from_pretrained(code_embedding_model)

    self.comment_embedding_model = TFAutoModel.from_config(AutoConfig.from_pretrained(
      comment_embedding_model,
      hidden_size=self.hidden_size,
    ))
    self.comment_embedding_tokenizer = AutoTokenizer.from_pretrained(comment_embedding_model)

  def from_pairs(self, pairs: List[MongoDbPairDoc], batch_size=100) -> Iterator[EmbeddingPairBatch]:
    for batch_pairs in more_itertools.chunked(pairs, batch_size):
      codes = [self.__pre_process_tokens(pair['code_tokens']) for pair in batch_pairs]
      comments = [self.__pre_process_tokens(pair['comment_tokens']) for pair in batch_pairs]

      codes_embeddings = self.from_sentences(
        sentences=codes,
        model=self.code_embedding_model,
        tokenizer=self.code_embedding_tokenizer,
      )
      comments_embeddings = self.from_sentences(
        sentences=comments,
        model=self.comment_embedding_model,
        tokenizer=self.comment_embedding_tokenizer,
      )

      yield {
        "pairs_ids": [pair['id'] for pair in batch_pairs],
        "code_embeddings": codes_embeddings.numpy(),
        "comment_embeddings": comments_embeddings.numpy(),
      }

  def from_sentences(self, sentences: List[str], tokenizer, model):
    encoded_input = tokenizer(
      sentences, 
      padding='max_length', 
      max_length=self.embedding_max_length,
      truncation=True, 
      return_tensors='tf',
    )
    model_output = model(**encoded_input, return_dict=True)

    embeddings = self.__mean_pooling(model_output, encoded_input['attention_mask'])
    embeddings = tf.math.l2_normalize(embeddings, axis=1)
    return embeddings
  
  def __mean_pooling(self, model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = tf.cast(tf.tile(tf.expand_dims(attention_mask, -1), [1, 1, token_embeddings.shape[-1]]), tf.float32)
    return tf.math.reduce_sum(token_embeddings * input_mask_expanded, 1) / tf.math.maximum(tf.math.reduce_sum(input_mask_expanded, 1), 1e-9)
  
  def __pre_process_tokens(self, tokens) -> str:
    parsed = ' '.join(tokens).replace('\n', ' ')
    parsed = ' '.join(parsed.strip().split())
    return parsed
  