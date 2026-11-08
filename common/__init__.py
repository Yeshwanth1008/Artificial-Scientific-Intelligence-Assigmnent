import os

os.environ.setdefault("USE_TF", "0")  # never load TensorFlow (Keras 3 conflict on some machines)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
