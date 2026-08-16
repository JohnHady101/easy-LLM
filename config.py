import tiktoken

MODEL = "o3"
EMBEDDING_DIM = 128
VOCAB_SIZE = tiktoken.encoding_for_model(MODEL).n_vocab
BATCH_SIZE = 4
MAX_LENGTH = 8
STRIDE = 4
