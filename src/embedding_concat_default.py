import typing

import tensorflow as tf
from models import Embedding, EmbeddingConcat

class EmbeddingConcatDefault(EmbeddingConcat):
  def concatenate(self, code_embedding: Embedding, text_embedding: Embedding, reshape: tuple | None) -> Embedding:
    concatenated = typing.cast(tf.Tensor, tf.concat([code_embedding, text_embedding], axis=1))
    return tf.reshape(concatenated, shape=reshape) if reshape is not None else concatenated
