from dataclasses import dataclass
import random
from typing import List

import more_itertools
import tensorflow as tf

from models import CodeCommentPair, DatasetRepository, EmbeddingConcat, EmbeddingGenerator, PreProcesser, EmbeddingComparator
from utils import encoder_seq_len, encoder_hidden_size

@dataclass
class Training:
  dataset_repository: DatasetRepository[CodeCommentPair]
  pre_processer: PreProcesser
  embedding_generator: EmbeddingGenerator
  embedding_concat: EmbeddingConcat
  model: EmbeddingComparator
  batch_size: int
  training_samples_count: int
  negative_samples_count: int
  
  def run(self):
    training_dataset = self.dataset_repository.get_dataset()

    # TODO: Use negative_sample_count value here
    def generate_negative_samples(pairs: List[CodeCommentPair]):
      def get_random_index(exclude_index: int):
        pairs_len = len(pairs)
        index_list = [i for i in range(pairs_len) if i != exclude_index]
        return index_list[random.randint(0, len(index_list) - 2)]

      for index, pair in enumerate(pairs):
        negative_pair = pairs[get_random_index(index)]
        yield CodeCommentPair(
          id=pair.id,
          code_tokens=pair.code_tokens,
          comment_tokens=negative_pair.comment_tokens,
          partition=pair.partition,
          language=pair.language,
        )

    def gen_and_concat_embeddings(pairs: List[CodeCommentPair]):
      code_embedddings, text_embeddings = self.embedding_generator.from_code([self.pre_processer.process_code(pair.code_tokens) for pair in pairs]), self.embedding_generator.from_text([self.pre_processer.process_text(pair.comment_tokens) for pair in pairs])
      concatenated = self.embedding_concat.concatenate(code_embedddings, text_embeddings, reshape=(self.batch_size, -1))
      return concatenated

    def embeddings_dataset_generator():
      for group_pairs in more_itertools.grouper(training_dataset, self.batch_size, incomplete='ignore'):
        batch_pairs = list(group_pairs)
        batch_embeddings = gen_and_concat_embeddings(batch_pairs)
        yield (batch_embeddings, self.embedding_generator.target(1, self.batch_size))
        
        negative_samples = list(generate_negative_samples(batch_pairs))
        for batch_negative_pairs in more_itertools.chunked(negative_samples, self.batch_size):
          batch_negative_embeddings = gen_and_concat_embeddings(batch_negative_pairs)
          yield (batch_negative_embeddings, self.embedding_generator.target(0, self.batch_size))
    
    concat_embedding_spec = tf.TensorSpec(shape=(self.batch_size, encoder_seq_len * encoder_hidden_size * 2), dtype=tf.float64) # type: ignore
    target_spec = tf.TensorSpec(shape=(self.batch_size, ), dtype=tf.int32) # type: ignore
    embedding_dataset = tf.data.Dataset.from_generator(embeddings_dataset_generator, output_signature=(concat_embedding_spec, target_spec))

    self.model.fit(embedding_dataset, epochs=1, batch_count=int(self.training_samples_count / self.batch_size))
    self.model.save()
