import numpy as np

data = np.memmap(".gitignore/tokenizers/bpe/tr/bpe_tr_train.bin", dtype=np.uint16, mode="r")
print("Total tokens:", len(data))