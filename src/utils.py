def decode_tensor_string(tensor) -> str:
  return tensor.numpy().decode('utf-8')

random_seed = 42
encoder_seq_len = 512
encoder_hidden_size = 768

codesearchnet_dataset_len = {
  "test": {
    "java": 26909,
    "python": 22176
  },
  "train": {
    "java": 454451,
    "python": 412178
  },
  "valid": {
    "java": 15328,
    "python": 23107
  }
}