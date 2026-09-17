# Koi

A small hybrid language model, built from scratch. Every **odd** layer is a
**Gated DeltaNet implementation** (linear attention with a fixed size recurrent
state). Every **even** layer is **local dense attention** (causal sliding window
softmax attention with RoPE). The result swims through a 16,384 token context
without ever building a KV cache that grows with the sequence.

Named after the koi fish, which glides through delta water without losing a scale.

```text
layer 1  |  Gated DeltaNet   long range memory, fixed size state, linear cost
layer 2  |  Local dense      sliding window 1024, RoPE, crisp short range mixing
layer 3  |  Gated DeltaNet   ...
layer 4  |  Local dense      ...
   ...        ...
layer 12 |  Local dense
```


## Why hybrid

| | Gated DeltaNet layers | Local dense layers |
|---|---|---|
| State | one fixed matrix per head, `head_dim x head_dim` | last `window` keys + values |
| Cost | linear in context, O(1) per decoded token | O(window) per token |
| Good at | slow forgetting, long range association | precise nearby token mixing |
| Cache size at 16k ctx | constant | constant (`window` slots, ring buffer) |

Softmax attention alone is O(L^2) in memory and compute. Pure recurrent layers
get fuzzy at precise recall. Alternating the two keeps the best of both.

## The Gated DeltaNet layers

A from-scratch **Gated DeltaNet implementation**. Each layer keeps a matrix
state `S` of shape `head_dim x head_dim` per head and updates it every token
with the gated delta rule:

```text
S_t = S_{t-1} * diag(alpha_t) - beta_t * (diag(alpha_t) S_{t-1}^T k_t - v_t) (x) k_t
o_t = S_t^T q_t
```

- `alpha` is a channel wise decay gate: each state row forgets at its own speed
  (low rank projection, initialized to forget slowly).
- `beta` is the write strength: how much of the prediction error to commit.
- q/k/v pass through depthwise causal short convolutions, SiLU, and L2 norm.
- Training uses a chunkwise parallel form: big matmuls inside a chunk, one
  cheap scan across chunks (128 chunks for a 16k sequence).
- Inference uses the plain recurrence: O(1) per token, fixed memory.

## The local dense layers

Causal sliding window attention, width 1024, with rotary position embeddings.
Training runs blocked attention through `scaled_dot_product_attention`: each
query block of 128 tokens only loads the keys it is allowed to see, so no
`L x L` attention matrix is ever materialized. Inference keeps a preallocated
ring buffer of the last `window` keys and values per layer: new token slides
in, oldest token slides out, attention reads the whole pile in one shot.

## Training/inference

Download the source code, use cmd in the directory and run either:
`python -m koi.train`
OR:
`python -m koi.inference`

## Installation

```bash
git clone https://github.com/MyGoodNautilus/koi.git
cd koi
pip install -r requirements.txt
```

## Training Results
After a 5 hour training run, we achieved these results.
| Metric   | Value  |
|----------|--------|
| Parameters    | 126.3M |
| Loss     | ~1.4 |
| PPL      | ~3.9   |

## Output:
```
He said to all of them. "Good idea, when you look like you're giving a real-life bastard to you when you want. That's a huge plus. It's not a long-term life long run."
```
This output, while not traditionally excellent, I would regard it's definitely noticeably decent compared to earlier versions.

## Performance
~100 t/s at FP32 precision, with constant VRAM usage and nearly constant speed.

Using general scaling laws, Q5_K_M (~5.5 bpw) should get us upwards of ~600 t/s on my laptop GPU.

For comparison, gpt-2-small gets roughly 20 t/s on GPUs like mine (notably, with only a 1024-token context that explodes in VRAM). This suggests that the architecture provides substantial speed improvements while maintaining ~95% recall accuracy through attention-logit calculations.

A flagship data-center GPU has roughly 20.8× the raw bandwidth of my laptop GPU, which puts a theoretical high-end target around 12k t/s. If we translate that to something like an 18B model, the approximate MoE size of flagship models, that works out to roughly 80 t/s without requiring insane levels of optimization.

Take this all with a grain of salt, as these were taken linearly, and comparisons were made without assuming optimized attention mechanisms in either model. As future work goes on, I expect to double or triple t/s.


## Notes
- The Gated DeltaNet recurrence math runs in fp32 with bf16 autocast around it;
  blocked window attention runs in the autocast dtype with fp32 decode scores.
- Gradient checkpointing is on by default in training; a 12 layer, d_model 1024
  fish fits a 16k context in roughly 6-8 GB of VRAM.
- MIT licensed. Go fish.
- The training data sample uses my book. Free publicity.

## References

- Yang, Kautz, Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta
  Rule*. The base recurrent linear attention design.
- *Kimi Linear / KDA*. A Gated DeltaNet implementation with channel wise decay
  gates; the inspiration for the low rank alpha gate used here.
- Dao et al. *FlashAttention*, and Vaswani et al. *Attention Is All You Need*.
  The softmax attention baseline this hybrid tames with a sliding window.
- Su et al. *RoFormer (RoPE)*. Rotary position embeddings on the local layers.
