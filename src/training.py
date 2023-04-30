from dataclasses import dataclass
import random
from typing import List

import more_itertools
import tensorflow as tf
from embedding_generator_default import EmbeddingGeneratorDefault
from pre_processer_default import PreProcesserDefault

from embedding_comparator import EmbeddingComparator
from embedding_concat_default import EmbeddingConcatDefault
from json_parser import OrJsonParser
from models import CodeCommentPair, DatasetRepository, EmbeddingConcat, EmbeddingGenerator, PreProcesser
from training_dataset import TrainingDataset
from utils import encoder_seq_len, encoder_hidden_size

batch_size = 128
batch_count = 105
@dataclass
class Training:
  dataset_repository: DatasetRepository[CodeCommentPair]
  pre_processer: PreProcesser
  embedding_generator: EmbeddingGenerator
  embedding_concat: EmbeddingConcat
  model: EmbeddingComparator
  negative_samples_count: int = 1

  def run(self):
    training_dataset = self.dataset_repository.get_dataset()

    def generate_negative_samples(pairs: List[CodeCommentPair]):
      pairs_len = len(pairs)

      def get_random_index(exclude_index: int):
        index_list = [i for i in range(pairs_len) if i != exclude_index]
        return index_list[random.randint(0, len(index_list) - 2)]

      for index, pair in enumerate(pairs):
        negative_pair = pairs[get_random_index(index)]
        yield CodeCommentPair(
          id=pair.id,
          code_tokens=pair.code_tokens,
          comment_tokens=negative_pair.comment_tokens
        )

    def gen_and_concat_embeddings(pairs: List[CodeCommentPair]):
      code_embedddings, text_embeddings = self.embedding_generator.from_code([self.pre_processer.process_code(pair.code_tokens) for pair in pairs]), self.embedding_generator.from_text([self.pre_processer.process_text(pair.comment_tokens) for pair in pairs])
      concatenated = self.embedding_concat.concatenate(code_embedddings, text_embeddings, reshape=(batch_size, -1))
      return concatenated

    def embeddings_dataset_generator():
      for group_pairs in more_itertools.grouper(training_dataset, batch_size, incomplete='ignore'):
        batch_pairs = list(group_pairs)
        batch_embeddings = gen_and_concat_embeddings(batch_pairs)
        yield (batch_embeddings, self.embedding_generator.target(1, batch_size))
        
        negative_samples = list(generate_negative_samples(batch_pairs))
        for batch_negative_pairs in more_itertools.chunked(negative_samples, batch_size):
          batch_negative_embeddings = gen_and_concat_embeddings(batch_negative_pairs)
          yield (batch_negative_embeddings, self.embedding_generator.target(0, batch_size))
    
    concat_embedding_spec = tf.TensorSpec(shape=(batch_size, encoder_seq_len * encoder_hidden_size * 2), dtype=tf.float64) # type: ignore
    target_spec = tf.TensorSpec(shape=(batch_size, ), dtype=tf.int32) # type: ignore
    embedding_dataset = tf.data.Dataset.from_generator(embeddings_dataset_generator, output_signature=(concat_embedding_spec, target_spec))
    self.model.fit(embedding_dataset, batch_size=batch_size, epochs=1, steps_count=batch_count * (self.negative_samples_count + 1))
    # self.model.save()

    # DOING: check generated embeddings vs normalized embeddings (do I need to normalize?)

Training(
  dataset_repository=TrainingDataset(
    json_parser=OrJsonParser(), 
    samples_count=batch_size * batch_count,
  ),
  pre_processer=PreProcesserDefault(),
  model=EmbeddingComparator(),
  embedding_concat=EmbeddingConcatDefault(),
  embedding_generator=EmbeddingGeneratorDefault(), 
).run()
