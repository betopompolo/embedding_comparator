def decode_tensor_string(tensor) -> str:
  return tensor.numpy().decode('utf-8')

random_seed = 42
encoder_seq_len = 512
encoder_hidden_size = 768
