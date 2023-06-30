from models import Runnable
from utils import build_model


class ModelSummary(Runnable):
  def run(self):
    build_model(8).summary()