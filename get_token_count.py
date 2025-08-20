import numpy as np

data = np.memmap("tokenizers/bpe/en/bpe_en_train.bin", dtype=np.uint16, mode="r")
print("Total tokens:", len(data))
