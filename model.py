# model.py
# Decoder-only Transformer architecture for character-level language modeling

import tensorflow as tf

# ─────────────────────────────────────────────
# 1. Multi-Head Self Attention
# ─────────────────────────────────────────────
# Attention lets the model look at other characters in the sequence
# to understand context. "Multi-head" means we do this multiple times
# in parallel with different learned perspectives, then combine results.


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, n_embd, n_heads, block_size, dropout=0.2):
        super().__init__()

        # Make sure embedding size is evenly divisible by number of heads
        # e.g. 384 embedding / 6 heads = 64 per head
        assert n_embd % n_heads == 0

        self.n_heads = n_heads  # number of attention heads
        self.head_size = n_embd // n_heads  # size of each head
        self.n_embd = n_embd  # total embedding dimension

        # One big matrix to compute Q, K, V all at once (more efficient)
        # Q = what am I looking for?
        # K = what do I contain?
        # V = what do I actually pass forward if someone attends to me?
        self.qkv_proj = tf.keras.layers.Dense(3 * n_embd, use_bias=False)

        # Final linear layer to mix the outputs of all heads together
        self.out_proj = tf.keras.layers.Dense(n_embd, use_bias=False)

        # Dropout on attention weights — prevents relying too heavily on any one connection
        self.attn_dropout = tf.keras.layers.Dropout(dropout)

        # CHANGE 5: Added dropout after out_proj for fuller regularization on attention output
        self.resid_dropout = tf.keras.layers.Dropout(dropout)

        # Causal mask: prevents the model from seeing future characters
        # e.g. when predicting position 5, it should not see positions 6, 7, 8...
        # We build an upper-triangular matrix of -1 billion values
        # So when softmax is applied, those positions become ~0 (ignored)
        mask = 1 - tf.linalg.band_part(tf.ones((block_size, block_size)), -1, 0)
        self.causal_mask = mask * -1e9

    def call(self, x, training=False):
        # x shape: (Batch size, Sequence length, Embedding size)
        B, T, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        # CHANGE 2: Guard against sequences longer than block_size to avoid silent failures
        tf.debugging.assert_less_equal(
            T, self.causal_mask.shape[0], message="Sequence length exceeds block_size"
        )

        # Project input into Q, K, V all at once, then split into 3 equal parts
        qkv = self.qkv_proj(x)  # (B, T, 3 * n_embd)
        q, k, v = tf.split(qkv, 3, axis=-1)  # each becomes (B, T, n_embd)

        # Helper function: reshape (B, T, n_embd) → (B, n_heads, T, head_size)
        # This separates the embedding into individual heads
        def split_heads(t):
            t = tf.reshape(t, (B, T, self.n_heads, self.head_size))
            return tf.transpose(t, [0, 2, 1, 3])

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention score
        # dot product of Q and K tells us: how much should token i attend to token j?
        # We divide by sqrt(head_size) to keep gradients stable (prevents huge values)
        scale = tf.math.sqrt(tf.cast(self.head_size, tf.float32))
        scores = tf.matmul(q, k, transpose_b=True) / scale  # (B, heads, T, T)

        # Apply causal mask — adds -1 billion to future positions
        # so after softmax they become zero (model cannot cheat by looking ahead)
        scores = scores + self.causal_mask[:T, :T]

        # Softmax converts raw scores into probabilities (all sum to 1)
        weights = tf.nn.softmax(scores, axis=-1)  # (B, heads, T, T)

        # Dropout on attention weights
        weights = self.attn_dropout(weights, training=training)

        # Multiply attention weights by Values to get the final attended output
        out = tf.matmul(weights, v)  # (B, heads, T, head_size)

        # Merge all heads back together
        out = tf.transpose(out, [0, 2, 1, 3])  # (B, T, heads, head_size)
        out = tf.reshape(out, (B, T, self.n_embd))  # (B, T, n_embd)

        # Final linear projection to mix information across heads
        # CHANGE 5: Apply residual dropout after out_proj
        return self.resid_dropout(
            self.out_proj(out), training=training
        )  # (B, T, n_embd)


# ─────────────────────────────────────────────
# 2. Feed Forward Network
# ─────────────────────────────────────────────
# After attention, each token processes the gathered information
# independently through this small neural network.
# It expands to 4x the size, applies non-linearity, then shrinks back.
# This is where the model "thinks" about what it just attended to.


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, n_embd, dropout=0.2):
        super().__init__()

        # CHANGE 6: Replaced Sequential with explicit layers for cleaner training flag handling
        # CHANGE 1: Replaced ReLU with GELU — standard in GPT-2 onwards, better for language modeling
        self.fc1 = tf.keras.layers.Dense(4 * n_embd, activation="gelu")
        self.fc2 = tf.keras.layers.Dense(n_embd)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        # Expand → GELU → shrink → dropout
        x = self.fc1(x)
        x = self.fc2(x)
        return self.dropout(x, training=training)


# ─────────────────────────────────────────────
# 3. Transformer Block
# ─────────────────────────────────────────────
# One full transformer block = Attention + FeedForward
# with LayerNorm and residual connections around each.
#
# Residual connection means: output = input + what_we_learned
# LayerNorm is applied BEFORE each sub-layer ("pre-norm" style)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, n_embd, n_heads, block_size, dropout=0.2):
        super().__init__()
        self.attention = MultiHeadSelfAttention(n_embd, n_heads, block_size, dropout)
        self.ffn = FeedForward(n_embd, dropout)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False):
        # Normalize → Attend → Residual
        x = x + self.attention(self.norm1(x), training=training)
        # Normalize → FeedForward → Residual
        x = x + self.ffn(self.norm2(x), training=training)
        return x


# ─────────────────────────────────────────────
# 4. Full Transformer Model
# ─────────────────────────────────────────────
# Characters → Embeddings → N Transformer Blocks → Output probabilities
#
# Token embedding: converts each character ID into a vector of numbers
# Positional embedding: tells the model WHERE in the sequence each token is


class ShakespeareTransformer(tf.keras.Model):
    def __init__(self, vocab_size, n_embd, n_heads, n_layers, block_size, dropout=0.2):
        super().__init__()

        self.block_size = block_size  # maximum sequence length the model can handle

        # Lookup table: converts each character ID → a vector of size n_embd
        self.token_emb = tf.keras.layers.Embedding(vocab_size, n_embd)

        # Lookup table: converts each position (0,1,2,...) → a vector of size n_embd
        self.pos_emb = tf.keras.layers.Embedding(block_size, n_embd)

        # Stack of N transformer blocks — deeper = more powerful
        self.blocks = [
            TransformerBlock(n_embd, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ]

        # Final LayerNorm before the output head
        self.norm = tf.keras.layers.LayerNormalization()

        # CHANGE 4: lm_head no longer has its own weight matrix.
        # Instead it reuses the token embedding weights (weight tying).
        # This reduces parameters and often improves performance (GPT-2 style).
        # We store vocab_size and n_embd to manually compute logits in call().
        self.vocab_size = vocab_size
        self.n_embd = n_embd

        # Output bias (optional but common practice even with weight tying)
        self.lm_bias = self.add_weight(
            name="lm_bias", shape=(vocab_size,), initializer="zeros", trainable=True
        )

    def call(self, x, training=False):
        # x shape: (Batch, Sequence Length) — integer token IDs
        B, T = tf.shape(x)[0], tf.shape(x)[1]

        # Convert token IDs to vectors: (B, T) → (B, T, n_embd)
        tok = self.token_emb(x)

        # Create position indices [0, 1, 2, ..., T-1] and embed them
        pos = self.pos_emb(tf.range(T))  # (T, n_embd)

        # Add token and position embeddings — now each vector knows WHAT and WHERE
        x = tok + pos  # (B, T, n_embd)

        # Pass through each transformer block one by one
        for block in self.blocks:
            x = block(x, training=training)

        # Final normalization
        x = self.norm(x)

        # CHANGE 4: Weight-tied output projection
        # Multiply by the transposed token embedding matrix instead of a separate Dense layer
        # embedding matrix shape: (vocab_size, n_embd) → transpose → (n_embd, vocab_size)
        emb_matrix = self.token_emb.embeddings  # (vocab_size, n_embd)
        logits = (
            tf.matmul(x, emb_matrix, transpose_b=True) + self.lm_bias
        )  # (B, T, vocab_size)

        return logits

    # ─────────────────────────────────────────────
    # CHANGE 7: generate() method for autoregressive text generation
    # ─────────────────────────────────────────────
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=40, top_p=0.9):
        """
        Autoregressively generate new tokens given a seed sequence.

        Args:
            idx           : (1, T) integer tensor — seed token IDs
            max_new_tokens: how many new characters to generate
            temperature   : controls randomness (lower = more deterministic)
            top_k         : only sample from top K most likely characters (0 = disabled)
            top_p         : only sample from top characters summing to probability P (0 = disabled)
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size so positional embeddings don't go out of range
            idx_cond = idx[:, -self.block_size :]

            # Forward pass — get logits for the last token position
            logits = self(idx_cond, training=False)  # (1, T, vocab_size)
            logits = logits[:, -1, :] / temperature  # (1, vocab_size)

            # ── Top-K filtering ──
            # Keep only the top_k highest scoring characters
            # Set everything else to -infinity so they can't be sampled
            if top_k > 0:
                # Get the top_k values and their indices
                top_k_values, _ = tf.math.top_k(logits, k=top_k)
                # The minimum value in the top_k set
                min_top_k = top_k_values[:, -1:]  # (1, 1)
                # Zero out anything below that threshold
                logits = tf.where(
                    logits < min_top_k, tf.fill(tf.shape(logits), -1e9), logits
                )

            # ── Top-P (nucleus) filtering ──
            # Sort logits descending, compute cumulative softmax probabilities,
            # and remove tokens once cumulative probability exceeds top_p
            if top_p > 0.0:
                # Convert logits to probabilities
                probs = tf.nn.softmax(logits, axis=-1)  # (1, vocab_size)

                # Sort probabilities in descending order
                sorted_indices = tf.argsort(
                    probs, direction="DESCENDING", axis=-1
                )  # (1, vocab_size)
                sorted_probs = tf.gather(
                    probs, sorted_indices, batch_dims=1
                )  # (1, vocab_size)

                # Cumulative sum of sorted probabilities
                cumulative_probs = tf.cumsum(
                    sorted_probs, axis=-1, exclusive=True
                )  # (1, vocab_size)

                # Remove tokens where cumulative prob already exceeds top_p
                sorted_mask = cumulative_probs >= top_p
                # Apply mask — set those logits to -inf
                sorted_logits = tf.gather(logits, sorted_indices, batch_dims=1)
                sorted_logits = tf.where(
                    sorted_mask, tf.fill(tf.shape(sorted_logits), -1e9), sorted_logits
                )

                # Scatter back to original order
                original_order = tf.argsort(sorted_indices, axis=-1)
                logits = tf.gather(sorted_logits, original_order, batch_dims=1)

            # Sample from the filtered distribution
            next_id = tf.random.categorical(
                logits, num_samples=1, dtype=tf.int32
            )  # (1, 1)

            # Append the new token to the sequence
            idx = tf.concat([idx, next_id], axis=1)  # (1, T+1)

        return idx


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Hyperparameters
    VOCAB_SIZE = 65  # 65 unique characters in TinyShakespeare
    N_EMBD = 384  # size of each token's embedding vector
    N_HEADS = 6  # number of attention heads (384 / 6 = 64 per head)
    N_LAYERS = 6  # number of stacked transformer blocks
    BLOCK_SIZE = 256  # max sequence length (context window)
    DROPOUT = 0.2  # 20% of connections randomly dropped during training
    BATCH_SIZE = 2  # small batch just for testing

    # Build the model
    model = ShakespeareTransformer(
        VOCAB_SIZE, N_EMBD, N_HEADS, N_LAYERS, BLOCK_SIZE, DROPOUT
    )

    # Create a dummy input — 2 sequences of 256 character IDs (all zeros for now)
    dummy_input = tf.zeros((BATCH_SIZE, BLOCK_SIZE), dtype=tf.int32)

    # Forward pass through the model
    logits = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"\nModel built successfully!")
    model.summary()

    # Test generate()
    seed = tf.zeros((1, 1), dtype=tf.int32)
    generated = model.generate(seed, max_new_tokens=50)
    print(f"\nGenerate test — output shape: {generated.shape}")  # (1, 51)
