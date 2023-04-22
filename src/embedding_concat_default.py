import typing

import tensorflow as tf
from models import EmbeddingConcat


class EmbeddingConcatDefault(EmbeddingConcat):
  def concatenate(self, code_embedding, text_embedding):
    return typing.cast(tf.Tensor, tf.concat([code_embedding, text_embedding], axis=1))