from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable

import tensorflow as tf
from embedding_generator import EmbeddingGenerator

from models import CodeCommentPairRepository, DatasetType, SupportedCodeLanguages


@dataclass
class GenerateEmbeddingsDataset:
    programming_language: SupportedCodeLanguages
    dataset_type: DatasetType
    batch_size: int
    embedding_generator: EmbeddingGenerator
    pair_repository: CodeCommentPairRepository

    def write(self, on_write: Callable[[int], Any]):
        with tf.io.TFRecordWriter(self._get_dataset_path()) as writer:
            for batch_pairs in islice(self.pair_repository.batch(self.programming_language, self.dataset_type, batch_size=self.batch_size), 5):
                input, target = self.embedding_generator.generate(
                    batch_pairs), self.embedding_generator.generate_target(batch_pairs)

                serialized_input = tf.io.serialize_tensor(input).numpy()
                serialized_target = tf.io.serialize_tensor(target).numpy()
                feature_input = tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[serialized_input])
                )
                feature_target = tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[serialized_target])
                )

                features = {
                    'input': feature_input,
                    'target': feature_target,
                }
                example = tf.train.Example(
                    features=tf.train.Features(feature=features))

                on_write(self.batch_size)
                writer.write(example.SerializeToString())  # type: ignore

    def read(self):
        feature_description = {
            'input': tf.io.FixedLenFeature(shape=(self.batch_size, 786432), dtype=tf.string),
            'target': tf.io.FixedLenFeature(shape=(self.batch_size), dtype=tf.string),
        }

        def map_fn(example_proto):
            example = tf.io.parse_single_example(
                example_proto, feature_description)
            return (tf.io.parse_tensor(example['input'], out_type=tf.double), tf.io.parse_tensor(example['target'], out_type=tf.int32))

        dataset = tf.data.TFRecordDataset(self._get_dataset_path())
        return dataset.map(map_fn)

    def _get_dataset_path(self) -> str:
        return f'{self.programming_language}-{self.dataset_type}.tfrecords'
