# character level tokenizer and batch sampler for MiniShakespeare
import numpy as np
import tensorflow as tf

# read raw data text
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"length of dataset in characters: {len(text)}")

# build vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"vocabulary size: {vocab_size}")
print(f"characters: {''.join(chars)}")

# character <-> integer mappings
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [char_to_int[c] for c in s]


def decode(ids):
    return "".join([int_to_char[i] for i in ids])


# encode complete dataset
data = np.array(encode(text), dtype=np.int32)

# train and val split: 90% train, 10% val
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"train tokens: {len(train_data)}, \nval tokens: {len(val_data)}")


# batch sampler
def get_batch(split, block_size=256, batch_size=64):
    """
    Randomly sample a batch of (input, target) pairs.
    Target is input shifted by 1 - next character prediction.
    """
    source = train_data if split == "train" else val_data

    # randomly sample batch_size starting indices for the input
    ra = np.random.randint(0, len(source) - block_size, size=(batch_size))

    x = np.stack([source[i : i + block_size] for i in ra])
    y = np.stack([source[i + 1 : i + block_size + 1] for i in ra])

    return tf.constant(x), tf.constant(y)


# check
if __name__ == "__main__":
    x, y = get_batch("train")
    print(f"input batch shape: {x.shape}, \ntarget batch shape: {y.shape}")
    print(f"input batch (first example): {decode(x[0].numpy().tolist())}")
