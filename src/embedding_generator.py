from dataclasses import dataclass
from typing import List
import numpy as np
import tensorflow as tf
from models import CodeCommentPair, EmbeddingModel
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


# TODO: Pass only config and initialize model/tokenizers lazily?
@dataclass
class EmbeddingGenerator:
    text_embedding_model: EmbeddingModel
    code_embedding_model: EmbeddingModel
    text_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    code_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast

    def generate(self, pairs: List[CodeCommentPair]) -> tf.Tensor:
        text_embeddings = self.generate_comment(pairs)
        code_embeddings = self.generate_code(pairs)

        inputs = self.concat_embeddings(text_embeddings, code_embeddings)

        return inputs

    def generate_target(self, batch_pairs: List[CodeCommentPair]) -> tf.Tensor:
        target_embedding = tf.convert_to_tensor([pair.target for pair in batch_pairs])

        return target_embedding

    def generate_target_temp(self, target: int, batch_size: int) -> tf.Tensor:
        target_embedding = tf.convert_to_tensor([target for _ in range(batch_size)])

        return target_embedding

    # TODO: Replace this with generate_code (use generate_text as reference)
    def generate_code(self, pairs: List[CodeCommentPair]) -> tf.Tensor:
        code_embeddings = self._get_embeddings(
            [pair.code for pair in pairs],
            tokenizer=self.code_tokenizer,
            model=self.code_embedding_model,
        )

        return code_embeddings

    # TODO: Replace this with generate_text
    def generate_comment(self, pairs: List[CodeCommentPair]) -> tf.Tensor:
        text_embeddings = self._get_embeddings(
            [pair.comment for pair in pairs],
            tokenizer=self.text_tokenizer,
            model=self.text_embedding_model,
        )

        return text_embeddings

    def generate_text(self, texts: List[str]) -> tf.Tensor:
        text_embeddings = self._get_embeddings(
            texts,
            tokenizer=self.text_tokenizer,
            model=self.text_embedding_model,
        )

        return text_embeddings

    def concat_embeddings(
        self, text_embedding: tf.Tensor, code_embedding: tf.Tensor
    ) -> tf.Tensor:
        batch_size = self._get_batch_size_from_tensor(text_embedding)
        text_numpy: np.ndarray = text_embedding.numpy()  # type: ignore
        code_numpy: np.ndarray = code_embedding.numpy()  # type: ignore

        embedding_ndim = 3

        if text_numpy.ndim != embedding_ndim:
            raise ValueError(
                f"expecting ndim {embedding_ndim} for text_embedding, got {text_numpy.ndim}"
            )

        if code_numpy.ndim != embedding_ndim:
            raise ValueError(
                f"expecting ndim {embedding_ndim} for code_embedding, got {code_numpy.ndim}"
            )

        text_numpy = text_numpy.reshape((batch_size, -1))
        code_numpy = code_numpy.reshape((batch_size, -1))

        concatenated = np.empty(
            (batch_size, text_numpy.shape[1] * 2),
            dtype=object,
        )
        for row in range(text_numpy.shape[0]):
            concatenated[row] = np.concatenate(
                (text_numpy[row], code_numpy[row]),
                axis=None,
            )

        return tf.convert_to_tensor(concatenated, dtype=tf.float64)

    def _get_batch_size_from_tensor(self, tensor: tf.Tensor) -> int:
        return [dim for dim in tensor.shape.as_list() if dim is not None][0]

    def _get_embeddings(
        self,
        data: str | list[str],
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        model: EmbeddingModel,
    ) -> tf.Tensor:
        input = tokenizer(
            data, return_tensors="tf", padding="max_length", truncation=True
        )
        output = model(**input)
        return output.last_hidden_state
