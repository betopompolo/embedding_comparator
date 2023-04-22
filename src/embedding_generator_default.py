from dataclasses import dataclass

import tensorflow as tf
from models import Embedding, EmbeddingGenerator, EmbeddingModel, Tokenizer
from utils import encoder_seq_len


@dataclass
class EmbeddingGeneratorDefault(EmbeddingGenerator):
  text_embedding_model: EmbeddingModel
  code_embedding_model: EmbeddingModel
  text_tokenizer: Tokenizer
  code_tokenizer: Tokenizer
  embeddings_count = 0
  
  def from_text(self, text: str | list[str]) -> Embedding:
    text_embeddings = self._get_embeddings(
      text,
      tokenizer=self.text_tokenizer,
      model=self.text_embedding_model,
    )
    return text_embeddings
  
  def from_code(self, code: str | list[str]) -> Embedding:
    code_embeddings = self._get_embeddings(
      code,
      tokenizer=self.code_tokenizer,
      model=self.code_embedding_model,
    )
    return code_embeddings
  
  def target(self, target: int, batch_size: int) -> tf.Tensor:
    target_embedding = tf.convert_to_tensor([target for _ in range(batch_size)])

    return target_embedding

  def _get_embeddings(
    self,
    data: str | list[str],
    tokenizer: Tokenizer,
    model: EmbeddingModel,
  ) -> Embedding:
    input = tokenizer(
      data, 
      return_tensors="tf", 
      padding="max_length", 
      truncation=True, 
      max_length=encoder_seq_len,
    )
    output = model(**input)
    embedding = output.last_hidden_state
    (normalized_embedding, _) = tf.linalg.normalize(embedding, axis=1)
    return normalized_embedding
