from datasets import load_from_disk

train_pairs = load_from_disk('datasets/python_train/')

def batch_mapping(batch):
  return batch

for x in train_pairs.map(batch_mapping, batch_size=100, batched=True):
  pass