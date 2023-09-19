import os
from typing import Dict, cast
from simple_term_menu import TerminalMenu
import logging
from transformers import logging as transformers_logging
import tensorflow as tf
from create_embedding_db import CreateEmbeddingDb

from create_mongo_db import CreateMongoDb
from cs_net_validation import CSNetValidation
from models import build_dense_model, build_siamese_model, dual_encoder_model, multilayer_raw
from data_analysis import DataAnalysis
from runnable import Runnable
from train import Train


def disable_lib_logs():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel(logging.ERROR)
    tf.autograph.set_verbosity(0)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    transformers_logging.set_verbosity_error()


def show_menu(menu: Dict[str, Runnable]):
    terminal_menu = TerminalMenu(
        title="Select an option", menu_entries=menu.keys())
    selected_option_index: int = cast(int, terminal_menu.show())
    assert isinstance(selected_option_index, int), "Invalid option"

    option_handler_key = list(menu.keys())[selected_option_index]
    menu[option_handler_key].run()


def run():
    disable_lib_logs()
    show_menu({
        "Create Mongo database": CreateMongoDb(pair_filters=[
            {"language": 'python', "partition": 'test'},
            {"language": 'python', "partition": 'train'},
            {"language": 'python', "partition": 'valid'},
        ]),
        "Create Embeddings database": CreateEmbeddingDb(filters=[
            {"language": 'python', 'partition': 'train', 'count': 10000},
            {"language": 'python', 'partition': 'test', 'count': 2000},
            {"language": 'python', 'partition': 'valid', 'count': 2000},
        ]),
        "Train": Train(
            model=build_dense_model(4, 'dense_4'),
            train_count=20000,
            valid_count=4000,
            embeddings_dataset_name='embeddings_raw'
        ),
        "CSNet Validation": CSNetValidation(
            model_name='dense_2-20230903-171323',
        ),
        "Data Analysis": DataAnalysis(),
    })


if __name__ == "__main__":
    run()
