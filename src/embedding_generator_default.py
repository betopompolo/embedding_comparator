from dataclasses import dataclass

import tensorflow as tf
from models import Embedding, EmbeddingGenerator, EmbeddingModel, PreProcesser, Tokenizer
from pre_processer_default import PreProcesserDefault
from utils import encoder_seq_len, encoder_hidden_size

from transformers import AutoConfig, TFAutoModel, AutoTokenizer # type: ignore


code_config_default = AutoConfig.from_pretrained("microsoft/codebert-base", max_position_embeddings=encoder_seq_len, hidden_size=encoder_hidden_size)
text_config_default = AutoConfig.from_pretrained("bert-base-uncased", max_position_embeddings=encoder_seq_len, hidden_size=encoder_hidden_size)

@dataclass
class EmbeddingGeneratorDefault(EmbeddingGenerator):
  pre_processer: PreProcesser = PreProcesserDefault()
  text_embedding_model: EmbeddingModel = TFAutoModel.from_config(text_config_default)
  code_embedding_model: EmbeddingModel = TFAutoModel.from_config(code_config_default)
  text_tokenizer: Tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=text_config_default.max_position_embeddings)
  code_tokenizer: Tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", model_max_length=code_config_default.max_position_embeddings)
  
  def from_text(self, text: list[str]) -> Embedding:
    text_embeddings = self._get_embeddings(
      self.pre_processer.process_text(text),
      tokenizer=self.text_tokenizer,
      model=self.text_embedding_model,
    )
    return text_embeddings
  
  def from_code(self, code: list[str]) -> Embedding:
    code_embeddings = self._get_embeddings(
      self.pre_processer.process_code(code),
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
    return embedding
