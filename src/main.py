from training import Training
from query_validation import QueryValidation
from models import Runnable
from model_summary import ModelSummary
from import_cs_net_queries_to_mongo_db import ImportCSNetQueriesToMongoDb
from import_cs_net_pairs_to_mongo_db import ImportCSNetPairsToMongoDb
from experiment_parameters import ExperimentParameters
from cross_validation import CrossValidation
from typing import Dict
import random
import logging
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

experiments = [
    # ExperimentParameters(
    #   name="experiment_1",
    #   num_hidden_layers=2,
    #   programming_languages=['java', 'python'],
    # ),
    # ExperimentParameters(
    #   name="experiment_2",
    #   num_hidden_layers=4,
    #   programming_languages=['java', 'python'],
    # ),
    # ExperimentParameters(
    #   name="experiment_3",
    #   num_hidden_layers=8,
    #   programming_languages=['java', 'python'],
    # ),
    ExperimentParameters(
        name="experiment_4",
        num_hidden_layers=2,
        programming_languages=['python'],
    ),
    # ExperimentParameters(
    #   name="experiment_5",
    #   num_hidden_layers=4,
    #   programming_languages=['python'],
    # ),
    # ExperimentParameters(
    #   name="experiment_6",
    #   num_hidden_layers=8,
    #   programming_languages=['python'],
    # ),
]

# TODO: Start here when removing unused files
mode_dict: Dict[str, Runnable] = {
  'valid': QueryValidation(
    experiments
  ),
  'kvalid': CrossValidation(
    experiments
  ),
  'summary': ModelSummary(),
  'train': Training(
    experiments
  ),
  'import_cs_net_pairs': ImportCSNetPairsToMongoDb(),
  'import_cs_net_queries': ImportCSNetQueriesToMongoDb(),
}
mode = os.getenv('MODE')
mode = mode.lower() if mode != None else ''

if mode not in mode_dict.keys():
  raise ValueError(f'Invalid value for MODE env. Possible values are {list(mode_dict.keys())}')

mode_dict[mode].run()
