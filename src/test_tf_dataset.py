import os
import random
from typing import List
import more_itertools

from embedding_comparator import EmbeddingComparator

from embedding_generator import EmbeddingGenerator
from models import CodeCommentPair
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import orjson
from transformers import AutoTokenizer, TFAutoModel, logging, AutoConfig
logging.set_verbosity_error()

"""
Variables
"""
negative_samples_count = 1
batch_size = 64
seq_len = 512
hidden_size = 768

"""
Open dataset files (from CodeSearchNet)
"""
dataset_len = {
    "test": {
        "java": 26880,
        "python": 22144
    },
    "train": {
        "java": 256,
        "python": 256
    },
    "valid": {
        "java": 15296,
        "python": 23104
    }
}
dataset_type = 'train'
dataset_samples_count = sum(dataset_len[dataset_type].values())

train_files = tf.data.Dataset.list_files(os.path.join(os.getcwd(), 'datasets', 'train', '*.jsonl'))
lines = tf.data.TextLineDataset(train_files)

"""
Initialize embedding generator
"""
code_config = AutoConfig.from_pretrained("microsoft/codebert-base", max_position_embeddings=seq_len)
text_config = AutoConfig.from_pretrained("bert-base-uncased", max_position_embeddings=seq_len)
embedding_generator = EmbeddingGenerator(
    text_embedding_model=TFAutoModel.from_config(text_config),
    text_tokenizer= AutoTokenizer.from_pretrained("bert-base-uncased", model_max_length=text_config.max_position_embeddings),
    code_embedding_model=TFAutoModel.from_config(code_config),
    code_tokenizer=AutoTokenizer.from_pretrained("microsoft/codebert-base", model_max_length=code_config.max_position_embeddings),
)

"""
Generating an embeddings dataset from the CodeSearchNet dataset
"""
def map_pair_from_jsonl(jsonl: str) -> CodeCommentPair:
    def parse_tokens(tokens: list[str]) -> str:
        parsed=' '.join(tokens).replace('\n',' ')
        parsed=' '.join(parsed.strip().split())
        return parsed
    
    json = orjson.loads(jsonl)
    pair = CodeCommentPair(
        id=json['url'],
        code=parse_tokens(json['code_tokens']),
        comment=parse_tokens(json['docstring_tokens']),
    )

    return pair

def generate_negative_samples(pairs: List[CodeCommentPair]):
    def get_random(start, end, exclude, attempts=100):
        for _ in range(attempts):
            random_value = random.randint(start, end)
            if random_value != exclude:
                return random_value
        raise ValueError("too many attempts")
    
    pairs_len = len(pairs)
    negative_samples: List[CodeCommentPair] = []

    for i, _ in enumerate(pairs):
        negative_samples_indexes = [get_random(
            0, pairs_len - 1, i) for _ in range(negative_samples_count)]
        for negative_index in negative_samples_indexes:
            pair = pairs[negative_index]
            negative_samples.append(pair)
            
    return negative_samples

def get_code_comment_embeddings(pairs: List[CodeCommentPair], target: int):
    code_embedding, comment_embedding = embedding_generator.generate_code(pairs), embedding_generator.generate_comment(pairs)
    return tf.reshape(code_embedding, (batch_size, -1)), tf.reshape(comment_embedding, (batch_size, -1)), embedding_generator.generate_target_temp(target, batch_size), tf.constant([pair.id for pair in pairs], dtype=tf.string)

def embedding_dataset_gen():
    for batch_lines in lines.take(dataset_samples_count).batch(batch_size):
        pairs = [map_pair_from_jsonl(line.numpy().decode('utf-8')) for line in batch_lines]

        yield get_code_comment_embeddings(pairs, target=1)
        for negative_samples in more_itertools.chunked(generate_negative_samples(pairs), batch_size):
            yield get_code_comment_embeddings(negative_samples, target=0)

id_spec = tf.TensorSpec(shape=(None, ), dtype=tf.string) # type: ignore
embedding_spec = tf.TensorSpec(shape=(None, None), dtype=tf.float64) # type: ignore
target_spec = tf.TensorSpec(shape=(None, ), dtype=tf.int32) # type: ignore
embeddings_dataset = tf.data.Dataset.from_generator(embedding_dataset_gen, output_signature=(embedding_spec, embedding_spec, target_spec, id_spec))

"""
Write embeddings dataset in a .tfrecords files
"""
# def tensor_to_bytes(tensor):
#     value = tf.io.serialize_tensor(tensor).numpy()
#     """Returns a bytes_list from a string / byte."""
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def create_example(code_embedding, comment_embedding, target_embedding):
#     feature = {
#         "code": tensor_to_bytes(code_embedding),
#         "comment": tensor_to_bytes(comment_embedding),
#         "target": tensor_to_bytes(target_embedding),
#     }
#     return tf.train.Example(features=tf.train.Features(feature=feature))

# def parse_example(example):
#     feature_description = {
#         "code": tf.io.FixedLenFeature([], tf.string),
#         "comment": tf.io.FixedLenFeature([], tf.string),
#         "target": tf.io.FixedLenFeature([], tf.string),
#     }
#     example = tf.io.parse_single_example(example, feature_description)
#     code_embedding = tf.io.parse_tensor(example['code'], out_type=tf.double)
#     comment_embedding = tf.io.parse_tensor(example['comment'], out_type=tf.double)
#     target_embedding = tf.io.parse_tensor(example['target'], out_type=tf.int32)

#     return code_embedding, comment_embedding, target_embedding

# tf_dataset_path = os.path.join(os.getcwd(), f"{dataset_type}.tfrecords")
# with tf.io.TFRecordWriter(tf_dataset_path) as writer:
#     for code, comment, target in tqdm(embeddings_dataset):
#         example = create_example(code, comment, target)
#         writer.write(example.SerializeToString())

"""
Train the network using tfrecords
"""
def prepare_sample_for_training(code_emb, comment_emb, target_emb, _pairs_ids):
    input = tf.concat([code_emb, comment_emb], axis=1)

    return tf.ensure_shape(input, (None, seq_len * hidden_size * 2)), tf.ensure_shape(target_emb, (None, )) 

# tf_dataset = tf.data.TFRecordDataset(tf_dataset_path).map(parse_example).map(prepare_sample_for_training)
# model = EmbeddingComparator()
# model.fit(tf_dataset, batch_size=batch_size)
tf_dataset = embeddings_dataset.map(prepare_sample_for_training)
model = EmbeddingComparator()
model.fit(tf_dataset, batch_size=batch_size)
model.save()

"""
Make predictions with the model
"""
# def prepare_sample_for_prediction(code_emb, comment_emb, target_emb):
#     input = tf.concat([code_emb, comment_emb], axis=1)

#     return tf.ensure_shape(input, (None, seq_len * hidden_size * 2))

# model.load()
# test_dataset = embeddings_dataset.map(prepare_sample_for_prediction).take(1)
# for item in test_dataset:
#     print(model.predict(item))