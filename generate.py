# generate.py
# Author: Purvak Baliyan
# Load the trained Shakespeare Transformer and generate new text

import tensorflow as tf
import os

# Import our tokenizer (encode/decode) and the model architecture
from dataset import encode, decode, vocab_size
from model import ShakespeareTransformer

# ─────────────────────────────────────────────
# Auto Device Detection — GPU if available, else CPU
# ─────────────────────────────────────────────
# Check if any GPU is available on this machine (NVIDIA or Apple Metal)
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    # GPU found — TensorFlow will use it automatically
    # Enable memory growth so TF doesn't grab all GPU memory at once
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        # Memory growth must be set before GPUs are initialized
        # If it fails, TF will still use the GPU — just without the memory limit
        pass
    DEVICE = "/GPU:0"
    print(f"✅ GPU detected — using: {gpus[0].name}")
else:
    # No GPU found — fall back to CPU
    DEVICE = "/CPU:0"
    print("⚠️  No GPU detected — running on CPU (generation will be slower)")

# ─────────────────────────────────────────────
# Configuration — tweak these to change output
# ─────────────────────────────────────────────

# Path to the saved weights from training
WEIGHTS_PATH = "saved_model/weights.weights.h5"

# ─────────────────────────────────────────────
# SEED TEXT — this is where you control the input
# ─────────────────────────────────────────────
# The model will read this and continue writing from it.
# Best results come from Shakespeare character names or scene openings.
# The model was trained on Shakespeare — so the closer your seed is
# to something Shakespeare would write, the better the output.
#
# ✅ GOOD inputs (model knows these patterns well):
#   "ROMEO:"               → generates Romeo's dialogue
#   "HAMLET:"              → generates Hamlet's dialogue
#   "JULIET:"              → generates Juliet's dialogue
#   "KING RICHARD III:"    → generates a king's speech
#   "MACBETH:"             → generates Macbeth's dialogue
#   "ACT I\n"              → generates a full act opening
#   "My lord, I"           → generates a mid-speech continuation
#   "To be or not"         → continues the famous line in Shakespeare's style
#
# ❌ BAD inputs (model never saw these — output will be garbage):
#   "Hello my name is"     → modern English, not in training data
#   "The weather today"    → not Shakespearean
#   "XYZ123"               → random characters, no pattern to follow
#
# Rule: the more Shakespearean your seed, the better the output.
SEED_TEXT = "ROMEO:"

# How many new characters to generate after the seed
MAX_NEW_TOKENS = 500

# Temperature controls randomness of output
# 0.5 = focused and repetitive
# 0.8 = good balance (recommended)
# 1.2 = creative but sometimes nonsensical
TEMPERATURE = 0.8

# ← ADD THESE TWO LINES HERE
# Top-K: only sample from top 40 most likely characters (cuts out garbage)
TOP_K = 40
# Top-P: only sample from characters whose combined probability reaches 90%
TOP_P = 0.9


# Model hyperparameters — must match exactly what you used in train.py
VOCAB_SIZE = vocab_size  # 65 unique characters
N_EMBD = 512  # embedding size
N_HEADS = 8  # attention heads
N_LAYERS = 6  # transformer blocks
BLOCK_SIZE = 256  # context window
DROPOUT = 0.0  # no dropout during generation — we want deterministic embeddings

# ─────────────────────────────────────────────
# Load the model
# ─────────────────────────────────────────────

# Build the same model architecture we used during training
# Wrap inside tf.device so all ops run on the detected device
with tf.device(DEVICE):
    model = ShakespeareTransformer(
        vocab_size=VOCAB_SIZE,
        n_embd=N_EMBD,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT,
    )

    # Check that the weights file actually exists before trying to load
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"No weights found at '{WEIGHTS_PATH}'.\n"
            f"Please run 'python train.py' first to train the model."
        )

    # Do one forward pass with dummy input to initialize all weight shapes
    # TensorFlow needs this before it can load weights into the model
    dummy = tf.zeros((1, BLOCK_SIZE), dtype=tf.int32)
    model(dummy)

    # Load the trained weights from disk into the model
    model.load_weights(WEIGHTS_PATH)
    print(f"✅ Loaded weights from {WEIGHTS_PATH}")

# ─────────────────────────────────────────────
# Generate text
# ─────────────────────────────────────────────

print(f"\n{'─' * 50}")
print(f"  Device    : {DEVICE}")
print(f"  Seed      : {repr(SEED_TEXT)}")
print(f"  Length    : {MAX_NEW_TOKENS} characters")
print(f"  Temp      : {TEMPERATURE}")
print(f"  Top-K     : {TOP_K} | Top-P: {TOP_P}")
print(f"{'─' * 50}\n")

# Encode the seed text into token IDs
# e.g. "ROMEO:" → [18, 17, 13, 17, 12, 22] (list of integers)
seed_ids = encode(SEED_TEXT)

# Wrap in a tensor of shape (1, len(seed)) — the 1 is the batch dimension
seed_tensor = tf.constant([seed_ids], dtype=tf.int32)

# Run generation on the detected device
with tf.device(DEVICE):
    # Run the model's generate() method to autoregressively produce new tokens
    # It predicts one character at a time and feeds it back as input
    # TO THIS:
    generated = model.generate(
        seed_tensor,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
    )

# Decode the generated token IDs back into readable text
# generated[0] = first (and only) item in the batch
output_text = decode(generated[0].numpy().tolist())

# Print the full output — seed text + generated continuation
print(output_text)
print(f"\n{'─' * 50}")
print(f"  Generated {MAX_NEW_TOKENS} characters")
print(f"{'─' * 50}\n")
