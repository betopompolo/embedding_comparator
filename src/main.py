import os
from typing import Dict, cast
from simple_term_menu import TerminalMenu
import logging
from transformers import logging as transformers_logging
import tensorflow as tf
from create_embedding_db import CreateEmbeddingDb

from create_mongo_db import CreateMongoDb
from runnable import Runnable

def disable_lib_logs():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel(logging.ERROR)
    tf.autograph.set_verbosity(0)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    transformers_logging.set_verbosity_error()

def run():
    menu: Dict[str, Runnable] = {
        "Create Mongo database": CreateMongoDb(pair_filters=[
            { "language": 'python', "partition": 'test' },
            { "language": 'python', "partition": 'train' },
            { "language": 'python', "partition": 'valid' },
        ]),
        "Create Embeddings database": CreateEmbeddingDb(filters=[
            { "language": 'python', 'partition': 'train', 'count': 20000 },
            { "language": 'python', 'partition': 'test', 'count': 4000 },
            { "language": 'python', 'partition': 'valid', 'count': 4000 },
        ]),
    }

    terminal_menu = TerminalMenu(title="Select an option", menu_entries=menu.keys())
    selected_option_index: int = cast(int, terminal_menu.show())
    assert isinstance(selected_option_index, int), "Invalid option"
    
    option_handler_key = list(menu.keys())[selected_option_index]
    menu[option_handler_key].run()


if __name__ == "__main__":
    disable_lib_logs()
    run()