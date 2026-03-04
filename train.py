# Training loop for the Shakespeare Transformer

import tensorflow as tf
import numpy as np
import time
import os

# Import our custom dataset and model
# CHANGE 4: Moved 'encode' import to the top level — no more import inside function body
from dataset import get_batch, vocab_size, encode, decode
from model import ShakespeareTransformer

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
VOCAB_SIZE = vocab_size  # 65 unique characters
N_EMBD = 512  # size of each token's embedding vector
N_HEADS = 8  # number of attention heads (768 / 12 = 64 per head)
N_LAYERS = 6  # number of stacked transformer blocks
BLOCK_SIZE = 256  # how many characters the model sees at once
DROPOUT = 0.2  # randomly zero 20% of connections during training
BATCH_SIZE = 48  # number of sequences per training step
MAX_STEPS = 10000  # total number of training steps
EVAL_EVERY = 200  # print loss every N steps
EVAL_STEPS = 50  # how many batches to average when estimating loss
LEARN_RATE = 3e-4  # peak learning rate
WARMUP_STEPS = 100  # CHANGE 2: linearly ramp LR from 0 → peak over this many steps
GRAD_CLIP = 1.0  # CHANGE 1: max allowed global gradient norm
SAVE_DIR = "saved_model"  # final weights destination
CKPT_DIR = "checkpoints"  # CHANGE 5: mid-training checkpoint directory

# ─────────────────────────────────────────────
# CHANGE 2: Learning Rate Schedule (warmup + cosine decay)
# ─────────────────────────────────────────────
# Warmup prevents instability when weights are still random at the start.
# Cosine decay slowly reduces the LR toward the end for finer convergence.
_cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LEARN_RATE,
    decay_steps=MAX_STEPS - WARMUP_STEPS,
    alpha=0.1,  # floor: LR never drops below 10% of peak
)


def get_lr(step):
    """Linear warmup for the first WARMUP_STEPS, cosine decay after that."""
    if step < WARMUP_STEPS:
        return LEARN_RATE * (step + 1) / WARMUP_STEPS
    return float(_cosine_decay(step - WARMUP_STEPS))


# ─────────────────────────────────────────────
# Build the model
# ─────────────────────────────────────────────
model = ShakespeareTransformer(
    vocab_size=VOCAB_SIZE,
    n_embd=N_EMBD,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    block_size=BLOCK_SIZE,
    dropout=DROPOUT,
)

# Adam optimizer — LR is updated manually each step via get_lr()
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE)

# Create checkpoint and save directories up front
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Resume from checkpoint (if available)
# ─────────────────────────────────────────────
# Automatically finds the latest saved checkpoint in the checkpoints/ folder.
# This way, if training is interrupted, you can re-run the same script
# and it will continue from where it left off — no code changes needed.
def find_latest_checkpoint(ckpt_dir):
    """
    Scans the checkpoints/ folder and returns the highest step number
    that has a saved .weights.h5 file. Returns 0 if none found.
    """
    latest_step = 0
    if os.path.exists(ckpt_dir):
        for folder in os.listdir(ckpt_dir):
            # Checkpoint folders are named like: step_1000, step_2000, etc.
            if folder.startswith("step_"):
                step_num = int(folder.split("_")[1])
                ckpt_file = os.path.join(ckpt_dir, folder, "weights.weights.h5")
                # Only count it if the actual weights file exists
                if os.path.exists(ckpt_file):
                    latest_step = max(latest_step, step_num)
    return latest_step


# Find the latest checkpoint step
START_STEP = find_latest_checkpoint(CKPT_DIR)

if START_STEP > 0:
    # We found a checkpoint — load its weights
    ckpt_file = f"{CKPT_DIR}/step_{START_STEP}/weights.weights.h5"

    # We must do one forward pass first so TensorFlow initializes
    # all weight shapes before we try to load values into them
    dummy = tf.zeros((1, BLOCK_SIZE), dtype=tf.int32)
    model(dummy)

    # Load the saved weights from disk into the model
    model.load_weights(ckpt_file)
    print(f"✅ Resumed from checkpoint: step {START_STEP} ({ckpt_file})")
else:
    # No checkpoint found — start fresh from step 0
    print("No checkpoint found — starting from scratch.")


# ─────────────────────────────────────────────
# Loss Function
# ─────────────────────────────────────────────
def compute_loss(logits, targets):
    """
    Cross-entropy loss measures how wrong the model's predictions are.

    logits:  (B, T, vocab_size) — raw scores for every possible next character
    targets: (B, T)             — the actual correct next character IDs

    Lower loss = better predictions.
    """
    # CHANGE 3: Read vocab dimension from logits instead of using the VOCAB_SIZE global
    # This keeps the function self-contained and correct even if vocab ever changes
    vocab = tf.shape(logits)[-1]
    B_T = tf.shape(logits)[0] * tf.shape(logits)[1]

    # Flatten logits: (B, T, vocab_size) → (B*T, vocab_size)
    logits_flat = tf.reshape(logits, (B_T, vocab))

    # Flatten targets: (B, T) → (B*T,)
    targets_flat = tf.reshape(targets, (-1,))

    # sparse_categorical_crossentropy expects integer targets (not one-hot)
    # from_logits=True means we pass raw scores, not softmax probabilities
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        targets_flat, logits_flat, from_logits=True
    )

    return tf.reduce_mean(loss)


# ─────────────────────────────────────────────
# Estimate Loss (no weight updates)
# ─────────────────────────────────────────────
def estimate_loss():
    """
    Evaluate the model on both train and val sets without updating weights.
    Averages over EVAL_STEPS batches for a stable estimate.
    training=False disables dropout so results are deterministic.
    """
    results = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(EVAL_STEPS):
            x, y = get_batch(split, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)
            logits = model(x, training=False)
            loss = compute_loss(logits, y)
            losses.append(loss.numpy())
        results[split] = np.mean(losses)
    return results


# ─────────────────────────────────────────────
# Sample some text from the model
# ─────────────────────────────────────────────
def sample_text(seed_char="\n", max_new_tokens=200, temperature=0.8):
    """
    Generate a short preview of what the model can produce so far.
    Seeded with a single newline, temperature=0.8 gives slightly
    less random output than pure sampling.
    """
    seed_ids = tf.constant([encode(seed_char)], dtype=tf.int32)  # (1, 1)
    generated = model.generate(
        seed_ids, max_new_tokens=max_new_tokens, temperature=temperature
    )
    return decode(generated[0].numpy().tolist())


# ─────────────────────────────────────────────
# Single Training Step
# ─────────────────────────────────────────────
@tf.function  # compiles into a TF graph — makes training significantly faster
def train_step(x, y):
    """
    One full forward + backward pass:
    1. Forward  : model predicts next characters
    2. Loss     : measure how wrong the predictions are
    3. Gradients: figure out which direction to nudge each weight
    4. Clip     : cap gradient norm to prevent destabilizing large updates
    5. Update   : apply those nudges via optimizer
    """
    with tf.GradientTape() as tape:
        logits = model(x, training=True)  # dropout ON during training
        loss = compute_loss(logits, y)

    gradients = tape.gradient(loss, model.trainable_variables)

    # CHANGE 1: Clip gradients by global norm
    # If the combined magnitude of all gradients exceeds GRAD_CLIP,
    # every gradient is scaled down proportionally — prevents explosive updates
    gradients, _ = tf.clip_by_global_norm(gradients, GRAD_CLIP)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────
print("=" * 60)
print("       Shakespeare Transformer — Training")
print("=" * 60)
print(f"  Vocab size  : {VOCAB_SIZE}")
print(f"  Embedding   : {N_EMBD}")
print(f"  Heads       : {N_HEADS}")
print(f"  Layers      : {N_LAYERS}")
print(f"  Block size  : {BLOCK_SIZE}")
print(f"  Batch size  : {BATCH_SIZE}")
print(f"  Max steps   : {MAX_STEPS}")
print(f"  Start step  : {START_STEP}  (0 = fresh start, >0 = resumed)")
print(f"  Peak LR     : {LEARN_RATE}  (warmup {WARMUP_STEPS} steps → cosine decay)")
print(f"  Grad clip   : {GRAD_CLIP}")
print("=" * 60 + "\n")

start_time = time.time()

# Loop starts from START_STEP (0 if fresh, 4000 if resuming from checkpoint)
# This means if training crashes again, just re-run — it auto-resumes!
for step in range(START_STEP, MAX_STEPS):
    # CHANGE 2: Update optimizer's LR each step according to the schedule
    optimizer.learning_rate.assign(get_lr(step))

    # ── Periodic evaluation ──
    if step % EVAL_EVERY == 0:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        steps_sec = (step - START_STEP + 1) / elapsed if elapsed > 0 else 0.0
        current_lr = get_lr(step)

        print(
            f"Step {step:5d} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f} "
            f"| LR: {current_lr:.2e} | {steps_sec:.1f} steps/sec | {elapsed:.1f}s"
        )

        # Show a generated text sample every 1000 steps
        if step % 1000 == 0:
            print("\n--- Sample output ---")
            print(sample_text())
            print("---------------------\n")

    # CHANGE 5: Save a mid-training checkpoint every 1000 steps
    # If training crashes, resume from here instead of starting over
    if step > 0 and step % 1000 == 0:
        ckpt_path = f"{CKPT_DIR}/step_{step}/weights"
        os.makedirs(f"{CKPT_DIR}/step_{step}", exist_ok=True)
        model.save_weights(ckpt_path + ".weights.h5")
        print(f"  [Checkpoint saved → {ckpt_path}]")

    # ── Training step ──
    x, y = get_batch("train", block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)
    train_step(x, y)

# ─────────────────────────────────────────────
# CHANGE 6: Final evaluation now lives cleanly after the loop
# ─────────────────────────────────────────────
losses = estimate_loss()
total_time = time.time() - start_time

print(f"\nFinal     | Train: {losses['train']:.4f} | Val: {losses['val']:.4f}")
print(
    f"Total training time: {total_time:.1f}s  ({(MAX_STEPS - START_STEP) / total_time:.1f} steps/sec avg)"
)

print("\n── Final Sample (500 chars) ──")
print(sample_text(max_new_tokens=500))

# ─────────────────────────────────────────────
# Save final model weights
# ─────────────────────────────────────────────
model.save_weights(f"{SAVE_DIR}/weights.weights.h5")
print(f"\nFinal weights saved to {SAVE_DIR}/weights.weights.h5")
