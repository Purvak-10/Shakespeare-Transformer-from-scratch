# Shakespeare Transformer from Scratch

I built a GPT-style transformer from scratch in TensorFlow and trained it on Shakespeare. No Hugging Face, no pretrained weights, no existing implementations — just pure Python and math.

The model reads text one character at a time, learns the patterns, and generates new text that looks and feels like Shakespeare wrote it.

---

## Sample output

Seed: `ROMEO:`

```
ROMEO:
No remedy. Where is the provost?
MERCUTIO:
It is a parlous point to the flower: it is the
true of a world fellow; it is a sickness to me the wooer.
TYBALT:
Marry, I will not hear them for this afternoon.
ROMEO:
I am a sorry devilish to my love.
BENVOLIO:
Be not so right: it is a gentleman born.
MERCUTIO:
Any thing to seems with me; but that I will find
me hardly set down to be her mistress.
```

Seed: `to be or not`

```
to be or not proud?
BUCKINGHAM:
Have done, my lord; I have been ready.
GLOUCESTER:
Come, come to me; for you must die to-morrow.
KING EDWARD IV:
I swear I will take order from me to keep
The plainess of my soul I am plain.
CLARENCE:
I will not do't: and when I was gone to Brittany,
He shall not be accepted but a cause.
GLOUCESTER:
But I know not what you will, sir, what will you
Have ta'en no ground with me to a morrow;
And I, but not I; am content to my tale,
And let my dear success it.
```

Not perfect — but it learned "parlous", "ta'en", "to-morrow", character names,
dialogue format, and old English rhythm entirely on its own just by reading characters.

---

## What's inside

```
shakespeare-transformer/
├── data/
│   └── input.txt       # TinyShakespeare (~1.1M characters)
├── checkpoints/        # Auto-saved every 1000 steps during training
├── saved_model/        # Final trained weights
├── dataset.py          # Tokenizer and batch sampler
├── model.py            # Full transformer architecture
├── train.py            # Training loop
├── generate.py         # Text generation
├── evaluate.py         # Computes val loss and perplexity
└── requirements.txt    # Dependencies
```

---

## Architecture

Decoder-only transformer — same family as GPT — built entirely from scratch:

- ~25 million parameters
- 6 transformer blocks
- 8 attention heads (512 embedding / 8 = 64 per head)
- 512 embedding dimensions
- 256 character context window
- Causal self-attention with masking so it can't peek at future characters
- Layer norm + residual connections around every sub-layer
- Weight-tied output projection (GPT-2 style — output layer reuses embedding weights)
- Top-K + Top-P nucleus sampling during generation

Training details:
- Dataset: TinyShakespeare (~1.1M characters, 65 unique chars)
- Optimizer: Adam with linear warmup + cosine decay
- Gradient clipping at 1.0
- 10,000 steps, batch size 48, context length 256
- Final train loss: 0.6486 / val loss: 1.6882
- Validation perplexity: 5.41
- Trained on Apple M4 using tensorflow-metal

---

## Setup

```bash
git clone https://github.com/Purvak-10/Shakespeare-Transformer-from-scratch.git
cd Shakespeare-Transformer-from-scratch
```

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you're on a Mac with an M-series chip, add Metal for GPU acceleration:
```bash
pip install tensorflow==2.16.2 tensorflow-metal==1.2.0
```
Without it training runs on CPU and takes 10-15 hours. With Metal it's around 4-5 hours.

Download the dataset:
```bash
mkdir data
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

---

## Training

```bash
caffeinate -i python train.py
```

`caffeinate -i` keeps your Mac awake. On Linux/Windows just run `python train.py`.

It prints loss every 200 steps and shows a generated sample every 1000 steps so you
can watch the model improve in real time. Here's what to expect:

| Step  | Train Loss | What it looks like                      |
|-------|------------|-----------------------------------------|
| 0     | ~4.5       | Random characters, complete garbage     |
| 1000  | ~1.5       | Real words, basic Shakespeare structure |
| 3000  | ~1.2       | Sentences starting to form              |
| 5000  | ~1.1       | Coherent dialogue, character names      |
| 10000 | ~0.65      | Authentic Shakespeare style text        |

**If training crashes or gets interrupted, just re-run the same command.** The script
automatically scans the `checkpoints/` folder, finds the latest saved step, and
resumes from there. No manual changes needed.

---

## Generating text

```bash
python generate.py
```

---

## Evaluating the model

```bash
python evaluate.py
```

This computes validation loss and perplexity on the held-out 10% of the dataset.
Perplexity of 5.41 means the model is effectively choosing between ~5 equally likely
characters at each step out of 65 possible, it has eliminated ~92% of the uncertainty.

---

## Changing the input

Open `generate.py` and change `SEED_TEXT`:

```python
SEED_TEXT = "ROMEO:"
```

Character names work best — the model saw them thousands of times and has strong
patterns to follow:

```python
SEED_TEXT = "HAMLET:"
SEED_TEXT = "JULIET:"
SEED_TEXT = "MACBETH:"
SEED_TEXT = "OTHELLO:"
SEED_TEXT = "KING RICHARD III:"
```

Adding a few words after the name improves output even more:

```python
SEED_TEXT = "HAMLET:\nTo be or not"
SEED_TEXT = "ROMEO:\nWhat light"
SEED_TEXT = "to be or not"
```

Avoid modern English — the model never saw it:

```python
SEED_TEXT = "Hello my name is"    # won't work
SEED_TEXT = "The weather today"   # won't work
```

You can also control the output style:

```python
MAX_NEW_TOKENS = 500   # how many characters to generate
TEMPERATURE    = 0.8   # lower = repetitive, higher = creative, 0.8 is a good middle ground
TOP_K          = 40    # only consider the 40 most likely next characters
TOP_P          = 0.9   # nucleus sampling — cuts off unlikely characters
```

---

## Training on a different dataset

Swap `data/input.txt` with any `.txt` file you want. The tokenizer builds the
vocabulary automatically from whatever you give it.

If you already trained on Shakespeare and want to switch datasets, delete the old
checkpoints first — the vocabulary will be different and old weights won't load:

```bash
rm -rf checkpoints/ saved_model/
python train.py
```

---

## Changing model size

Edit these in `train.py`:

```python
N_EMBD     = 512   # embedding size — bigger = more expressive
N_HEADS    = 8     # attention heads — must divide evenly into N_EMBD
N_LAYERS   = 6     # transformer blocks — more = deeper reasoning
BLOCK_SIZE = 256   # context window — how many characters the model sees at once
MAX_STEPS  = 10000 # how long to train
BATCH_SIZE = 48    # lower this if you run out of memory
```

---

## Honest expectations

The model learns patterns, not meaning. It knows `ROMEO:` should be followed by
romantic dialogue because it saw that thousands of times — not because it understands
love. Individual sentences are mostly coherent but it loses track of logic across
longer stretches. That's just the nature of a small character-level model.

For a ~25M parameter model trained on 1MB of text on a laptop, the output is
surprisingly good.
