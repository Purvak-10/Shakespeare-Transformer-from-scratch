import math
import numpy as np
import tensorflow as tf
from dataset import get_batch, vocab_size
from model import ShakespeareTransformer

# ── Hyperparameters — must match train.py ──
VOCAB_SIZE = vocab_size
N_EMBD = 512
N_HEADS = 8
N_LAYERS = 6
BLOCK_SIZE = 256
BATCH_SIZE = 48

# ── Load model ──
model = ShakespeareTransformer(
    VOCAB_SIZE, N_EMBD, N_HEADS, N_LAYERS, BLOCK_SIZE, dropout=0.0
)
dummy = tf.zeros((1, BLOCK_SIZE), dtype=tf.int32)
model(dummy)
model.load_weights("saved_model/weights.weights.h5")


# ── Compute loss ──
def compute_val_loss(num_batches=100):
    losses = []
    for _ in range(num_batches):
        x, y = get_batch("val", block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)
        logits = model(x, training=False)
        vocab = tf.shape(logits)[-1]
        B_T = tf.shape(logits)[0] * tf.shape(logits)[1]
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            tf.reshape(y, (-1,)), tf.reshape(logits, (B_T, vocab)), from_logits=True
        )
        losses.append(tf.reduce_mean(loss).numpy())
    return np.mean(losses)


val_loss = compute_val_loss()
perplexity = math.exp(val_loss)

print(f"Val Loss   : {val_loss:.4f}")
print(f"Perplexity : {perplexity:.2f}")
