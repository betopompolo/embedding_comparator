import os
from embedding_comparator import EmbeddingComparator

from embedding_generator import EmbeddingGenerator
from models import CodeCommentPair
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import orjson
from transformers import AutoTokenizer, TFAutoModel

json_lines_count = 300

lines = tf.data.TextLineDataset([
    f'python_train_{json_lines_count}.jsonl'
])

embedding_generator = EmbeddingGenerator(
    text_embedding_model=TFAutoModel.from_pretrained(
        "bert-base-uncased"),
    text_tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
    code_embedding_model=TFAutoModel.from_pretrained(
        "microsoft/codebert-base", ),
    code_tokenizer=AutoTokenizer.from_pretrained(
        "microsoft/codebert-base"),
)

def get_embeddings():
    pairs = []
    for line in lines:
        line = line.numpy().decode('utf-8')
        json = orjson.loads(line)
        pair = CodeCommentPair(
            code=json['code'],
            comment=json['docstring'],
            target=1
        )
        pairs.append(pair)

    yield embedding_generator.generate(pairs), embedding_generator.generate_target(pairs)


input_spec = tf.TensorSpec(shape=(None, 786432), dtype=tf.float64)
target_spec = tf.TensorSpec(shape=(None, ), dtype=tf.int32)
dataset = tf.data.Dataset.from_generator(generator=get_embeddings, output_signature=(input_spec, target_spec)).batch(64)

model = EmbeddingComparator()
model.fit(dataset, batch_size=64)
