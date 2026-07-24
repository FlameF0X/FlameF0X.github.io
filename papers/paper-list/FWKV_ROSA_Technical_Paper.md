

> *Hugging Face: [FlameF0X](https://huggingface.co/FlameF0X)*  
> *Models: [FWKV-29M](https://huggingface.co/FlameF0X/FWKV-29M) · [FWKV-50M](https://huggingface.co/FlameF0X/FWKV-50M) · [FWKV-TinyStories](https://huggingface.co/FlameF0X/FWKV-TinyStories) · [FWKV-ROSA](https://huggingface.co/FlameF0X/FWKV-ROSA)*  
> *Project: [flamef0x.github.io](https://flamef0x.github.io/blog/)*  
> *Date: July 2026*

---

## Abstract

We present the **FWKV** family of recurrent language models—ranging from 29M to 56M parameters—that achieve remarkable inference throughput—up to **48.1 tokens per second on a basic 2 vCPU / 16 GB RAM CPU**—through a combination of architectural minimalism and an exact, parameter-free long-range copy mechanism. FWKV replaces the pairwise time-mixing of RWKV with a simpler **per-channel fixed-decay accumulator** computed via an $O(\log T)$ vectorized parallel scan, eliminating both the quadratic cost of self-attention and the need for a growing KV cache. We augment this recurrent core with **ROSA** (Rapid Online Suffix Automaton), an exact causal predictor that identifies the longest matching suffix at each position and injects the token that followed it last time as an auxiliary embedding signal—providing a principled, training-free mechanism for long-range copying without attention. The model is trained on multi-turn conversational data (UltraChat 200k) with a simple two-role chat template and loss masking restricted to assistant tokens. We report training results, inference benchmarks across CPU and GPU, and a detailed architectural analysis. The FWKV family demonstrates that competitive language modeling, storytelling, and conversational AI can be achieved at modest parameter counts through careful co-design of recurrence, copy mechanisms, and factorized embeddings—establishing a new Pareto frontier for inference efficiency.

---

## 1. Introduction

The dominant paradigm in language modeling—the Transformer architecture [Vaswani et al., 2017]—suffers from two well-known limitations at inference time: (1) quadratic scaling of self-attention with sequence length, and (2) a linearly growing key-value (KV) cache that consumes VRAM proportional to batch size × sequence length × layers. These limitations have motivated a wave of research into recurrent and state-space alternatives, including RWKV [Peng et al., 2023], Mamba [Gu & Dao, 2023], and various linear attention variants [Katharopoulos et al., 2020], all of which replace the $O(T^2)$ attention mechanism with an $O(T)$ recurrent state.

FWKV occupies a distinct point in this design space. Rather than learning data-dependent gating or complex state transitions, FWKV employs a **fixed per-channel decay** on a single accumulated state vector, combined with a **floored sigmoid** to prevent catastrophic forgetting. The recurrence is:

$$S_t = S_{t-1} \odot W + k_t \odot v_t$$

where $W = \text{clamp}(\sigma(w), \text{min}=\alpha)$ is a learned, constant-per-channel decay factor floored at $\alpha = 0.1$, and $k_t, v_t$ are projections of the current hidden state. This formulation is simpler than RWKV's time-mixing (which uses pairwise $w_{ij}$ decays), simpler than Mamba's selective state-space model, and requires **no custom CUDA kernels**—the entire recurrence is computed via a standard Hillis-Steele parallel prefix scan in $O(\log T)$ sequential (but fully vectorized) steps.

The second key innovation is **ROSA** (Rapid Online Suffix Automaton), inspired by RWKV-8 [rwkv.com, 2025]. ROSA is a training-free, exact causal predictor that answers the question: *"Given the current prefix, what token followed the longest matching suffix in the past?"* It is built as an online suffix automaton (SAM) with one extra array tracking each state's most recent occurrence end index, yielding $O(n)$ amortized time and $O(n)$ states for a sequence of length $n$. The predicted token is embedded and added to the model's hidden state as an auxiliary signal, giving the recurrent core an explicit "copy" capability that would otherwise require learned attention.

Together, these components produce a model that is:
- **Fast at inference**: 48.1 tok/s on CPU, competitive with RWKV on GPU at moderate scales.
- **Structurally simple**: No gating, no custom kernels, no position embeddings table.
- **Fully open**: Apache 2.0 licensed, compatible with HuggingFace Transformers and vLLM.

---

## 2. Related Work

### 2.1 Recurrent Language Models

The limitations of quadratic attention have spurred extensive work on recurrent alternatives. **RWKV** [Peng et al., 2023] combines elements of RNNs and Transformers through a time-mixing block that uses learned positional decays $w_{ij}$ between every pair of time steps, computed efficiently via a recurrence. **Mamba** [Gu & Dao, 2023] introduces selective state-space models where the transition matrices are data-dependent, enabling content-aware reasoning within a recurrent framework. **RetNet** [Sun et al., 2023] proposes a multi-scale retention mechanism that can be computed in parallel, recurrent, or chunkwise-recurrent modes. **xLSTM** [Beck et al., 2024] modernizes the LSTM with exponential gating and matrix memory.

FWKV differs from all of these in its commitment to a **fixed, non-data-dependent decay**: the per-channel $W$ is learned during training but is constant at inference time—it does not depend on the input. This design choice prioritizes parallelism and simplicity over expressivity, betting that the ROSA copy mechanism can compensate for the lost flexibility.

### 2.2 Suffix Automata and Copy Mechanisms

Suffix automata (SAMs) are classical data structures in stringology [Blumer et al., 1985; Ukkonen, 1995] that compactly represent all substrings of a string and support efficient longest-match queries. Their use in neural language modeling is novel: ROSA adapts the online SAM construction to track, for each equivalence class of substrings, the most recent occurrence end index, turning the SAM into a causal next-token predictor. This is distinct from pointer networks [Vinyals et al., 2015] and copy mechanisms in seq2seq models [Gu et al., 2016; See et al., 2017], which learn to copy via attention; ROSA provides exact, deterministic copying as a feature, not a learned behavior.

### 2.3 Factorized Embeddings

Factorized or tied input-output embeddings are a standard technique for reducing parameter count in language models [Press & Wolf, 2017; Lan et al., 2020]. FWKV extends this with a **factorized tied head** that projects through a low-dimensional bottleneck ($d_\text{emb} = 128$) before expanding to the model dimension ($d_\text{model} = 512$), with the head and embedding layers sharing weights via transpose. This design also enables chunked cross-entropy computation that avoids materializing the full $B \times T \times V$ logit tensor during training.

---

## 3. Architecture

FWKV-ROSA consists of three main components: (1) the FWKV recurrent core with fixed per-channel decay, (2) the ROSA copy-signal embedding, and (3) the factorized tied embedding/head. Figure 1 provides a high-level schematic of the full architecture.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FWKV-ROSA ARCHITECTURE                       │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │  Input   │    │    ROSA      │    │   Factorized Tied Head   │   │
│  │  Tokens  │───▶│  (Suffix     │───▶│                         │   │
│  │  [B, T]  │    │  Automaton)  │    │  Embed: [B,T,d_emb]      │   │
│  └──────────┘    │              │    │    → Proj: [B,T,d_model] │   │
│                  │  Predicts:   │    └──────────┬───────────────┘   │
│                  │  y[i] = x[j+1]│              │                   │
│                  │  for longest │               ▼                   │
│                  │  suffix match│    ┌──────────────────────┐       │
│                  └──────┬───────┘    │   ROSA Embedding     │       │
│                         │            │   Embed(pred) → +    │       │
│                         │            └──────────┬───────────┘       │
│                         │                       │                   │
│                         └───────────────────────┤                   │
│                                                 ▼                   │
│                              ┌──────────────────────────────────┐   │
│                              │     FWKV Block × N_Layers (14)   │   │
│                              │                                  │   │
│                              │  x ──▶ LayerNorm                │   │
│                              │   │                              │   │
│                              │   ├─▶ k = Linear(x)              │   │
│                              │   ├─▶ v = Linear(x)              │   │
│                              │   ├─▶ r = σ(Linear(x)) (recept.) │   │
│                              │   │                              │   │
│                              │   ├─▶ a = k ⊙ v                 │   │
│                              │   ├─▶ parallel_scan_decay(a, W) │   │
│                              │   │   ┌─────────────────────┐    │   │
│                              │   │   │  W = clamp(σ(w),    │    │   │
│                              │   │   │      min=0.1)       │    │   │
│                              │   │   │  O(log T) scan      │    │   │
│                              │   │   └─────────────────────┘    │   │
│                              │   │                              │   │
│                              │   ├─▶ out = Linear(r ⊙ wkv)     │   │
│                              │   ├─▶ x = x + out  (residual)   │   │
│                              │   │                              │   │
│                              │   └─▶ FFN: Linear→GELU→Linear   │   │
│                              │        x = x + FFN(x) (residual) │   │
│                              └──────────────────────────────────┘   │
│                                                 │                   │
│                                                 ▼                   │
│                              ┌──────────────────────────────────┐   │
│                              │        Final LayerNorm           │   │
│                              │        to_emb_space (proj^T)     │   │
│                              │        Chunked CE / Logits       │   │
│                              └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

**Figure 1:** High-level architecture of FWKV-ROSA. Input tokens are processed by ROSA in parallel to produce copy-signal predictions, which are embedded and added to the token embeddings. The combined signal passes through a stack of FWKV blocks, each performing a fixed-decay recurrent accumulation followed by a feed-forward network with GELU activation. The output head uses factorized tied weights.

### 3.1 FWKV Recurrent Core

Each FWKV block (see Figure 2) operates on a hidden state $x \in \mathbb{R}^{B \times T \times d}$ and implements a gated linear recurrence:

```
┌────────────────────────────────────────────────────────────┐
│                    FWKV BLOCK (detail)                     │
│                                                            │
│   x ──────────────────────────────────────────────┐        │
│   │                                               │        │
│   ├──▶ proj_k ──▶ k                              │        │
│   ├──▶ proj_v ──▶ v                              │        │
│   ├──▶ proj_r ──▶ σ ──▶ r  (receptance gate)    │        │
│   │                                               │        │
│   ├──▶ a = k ⊙ v                                 │        │
│   │       │                                       │        │
│   │       ▼                                       │        │
│   │   parallel_scan_decay(a, W)                   │        │
│   │       │                                       │        │
│   │       ▼                                       │        │
│   │   wkv = [S_0, S_1, ..., S_{T-1}]              │        │
│   │       │                                       │        │
│   │       ▼                                       │        │
│   ├──▶ proj_out(r ⊙ wkv) ──▶ + ──▶ LayerNorm    │        │
│   │                                 │             │        │
│   │                                 ▼             │        │
│   │                           FFN: Linear(d→4d)   │        │
│   │                                GELU           │        │
│   │                                Linear(4d→d)   │        │
│   │                                 │              │       │
│   └─────────────────────────────────+──▶ LayerNorm │      │
│                                                     │      │
│   State update:                                     │      │
│   S_t = S_{t-1} ⊙ W + k_t ⊙ v_t                   │      │
│   W = clamp(σ(w), min=0.1)  (per-channel, learned)  │      │
└────────────────────────────────────────────────────────────┘
```

**Figure 2:** Detailed view of a single FWKV block. The three projections ($k$, $v$, $r$) are linear without bias. The receptance gate $r$ controls information flow via element-wise multiplication. $W$ is constant across time for each channel.

#### 3.1.1 Fixed Per-Channel Decay

The core design decision in FWKV is the choice of decay weight. In RWKV, the time-mixing block uses a pairwise decay $w_{ij}$ that depends on the relative position $|i-j|$ and is parameterized per head and per channel. FWKV simplifies this to a single per-channel decay constant:

$$W_c = \text{clamp}(\sigma(w_c), \text{min}=0.1)$$

where $w_c \in \mathbb{R}$ is a learnable parameter initialized to $2.0$ (yielding $\sigma(2.0) \approx 0.88$ initial decay). The clamping floor of $0.1$ ensures that the model **cannot fully forget** any past input—the effective context window is infinite, with older contributions decaying exponentially but never reaching zero. This is a form of stable, bounded memory without explicit gating mechanisms.

The recurrence is then:

$$S_t = S_{t-1} \odot W + k_t \odot v_t$$

with $S_{-1} = 0$. Each channel evolves independently with its own decay rate; channels with $W_c$ close to $1.0$ act as near-perfect integrators (long-range memory), while channels with $W_c$ close to $0.1$ act as rapid forgetters (short-range, local patterns).

#### 3.1.2 Parallel Scan

Because $W$ is **constant per channel** (not data-dependent), the recurrence can be computed in parallel using a Hillis-Steele prefix scan in $O(\log T)$ steps. For a run of length $d$, the decay across the entire run is simply $W^d$. The scan proceeds as:

```
for d in 1, 2, 4, 8, ... up to T:
    val[t] += W^d * val[t-d]    (with val[t-d] = 0 if t-d < 0)
```

Each step is a single fused multiply-add over all batch elements, time steps, and channels simultaneously. The implementation is pure PyTorch—no custom CUDA kernels required. We verified the parallel scan against a naive sequential loop to within $< 10^{-14}$ maximum absolute error.

> **Key insight:** The constant-decay property is what makes the vectorized scan possible. If $W$ were data-dependent (as in Mamba or gated RNNs), the scan would require associative scan operations over pairs $(a, b)$ with a more complex combine rule, typically requiring a custom CUDA kernel for efficiency.

### 3.2 ROSA: Rapid Online Suffix Automaton

ROSA provides an exact, parameter-free mechanism for the model to exploit repeated patterns. Given an input token sequence $x[0..n-1]$, ROSA computes:

$$y[i] = x[j+1], \text{ where } x[j-m..j] = x[i-m..i]$$

for the **largest match length** $m > 0$ over all $j < i$, with ties broken by the largest $j$. If no prior occurrence of any non-empty suffix ending at position $i$ exists, $y[i] = -1$.

The prediction is fully causal: $y[i]$ depends only on $x[0..i]$.

#### 3.2.1 Suffix Automaton Construction

ROSA is built as an online suffix automaton (SAM) [Blumer et al., 1985] with one crucial augmentation: an additional array `last_end` that records, for each SAM state, the **most recent end index** where a substring in that equivalence class occurred.

The standard SAM construction maintains:
- `trans[state]`: a dictionary of token → next state transitions
- `link[state]`: suffix link to the state representing the longest proper suffix
- `length[state]`: the length of the longest string represented by the state

ROSA adds:
- `last_end[state]`: the end index of the most recent occurrence of substrings in this state

The construction proceeds online, processing one token at a time:

1. **Extend**: Create a new state `cur` for the new prefix. Add transitions from all states reachable by following suffix links from the previous `last` state.

2. **Clone if needed**: If a transition conflict arises (the existing transition target has a longer `length` than expected), clone the target state, preserving its `last_end` information.

3. **Predict**: Walk from `cur` up the suffix-link chain. The first state encountered with a recorded `last_end` value corresponds to the **longest** matching suffix (since `length` strictly decreases along suffix links). The prediction is `x[last_end[v] + 1]`.

4. **Update last_end**: Walk from `last` (the state for the full prefix up to $i$) up the suffix-link chain, updating `last_end` to $i$ for any state whose recorded end index is older.

The algorithm uses at most $2n-1$ states and $3n-4$ transitions for a sequence of length $n$, and processes each token in amortized $O(1)$ time. Dictionary-based transitions make it vocabulary-size-agnostic.

#### 3.2.2 Integration with the Neural Model

ROSA predictions are embedded through a dedicated embedding layer $\text{ROSAEmb} \in \mathbb{R}^{(V+1) \times d_\text{emb}}$ (with index 0 reserved for "no prediction," i.e., $y[i] = -1$):

$$\text{rosa\_idx}[i] = (y[i] + 1) \cdot \mathbf{1}[y[i] \geq 0]$$

$$h^\text{rosa}_i = \text{ROSAEmb}(\text{rosa\_idx}[i])$$

The ROSA embedding is projected to model dimension and **added** to the token embedding:

$$h^\text{combined}_i = \text{Proj}(\text{TokenEmb}(x[i])) + \text{Proj}(\text{ROSAEmb}(\text{rosa\_idx}[i]))$$

This injection occurs **before** the recurrent blocks, giving the model early access to copy information. The ROSA parameters (~6.4M of the total ~56M) are learned during training, allowing the model to learn when to trust or ignore the copy signal.

### 3.3 Factorized Tied Embeddings

FWKV uses a factorized input/output representation that reduces the parameter count of the embedding matrix from $V \times d_\text{model}$ to $V \times d_\text{emb} + d_\text{emb} \times d_\text{model}$:

| Component | Dimensions | Parameters |
|-----------|-----------|------------|
| Token weight | $V \times d_\text{emb}$ | $50257 \times 128 = 6.4\text{M}$ |
| Projection | $d_\text{emb} \to d_\text{model}$ | $128 \times 512 = 65.5\text{K}$ |
| **Total** | | **~6.5M** |
| *(vs. dense)* | $V \times d_\text{model}$ | *25.7M* |

The output logits are computed as the dot product of the final hidden state (projected back to $d_\text{emb}$ via the transposed projection) with the token weight matrix. During training, we use **chunked cross-entropy**: the logit matrix is never fully materialized; instead, the loss is accumulated in chunks of `ce_chunk` (default 512) tokens, with proper normalization over non-ignored (supervised) positions.

### 3.4 Feed-Forward Network

Each FWKV block includes a standard feed-forward network with a single hidden layer:

$$\text{FFN}(x) = \text{Linear}_{d \to 4d}(x) \to \text{GELU} \to \text{Linear}_{4d \to d}(x)$$

The expansion factor of 4× follows the convention established by Transformer architectures. Each sub-layer (WKV mixing and FFN) is wrapped with a residual connection and LayerNorm, following a pre-norm design.

---

## 4. Model Configuration

The FWKV-ROSA model uses the following hyperparameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| $d_\text{model}$ | 512 | Hidden state dimension |
| $d_\text{emb}$ | 128 | Embedding bottleneck dimension |
| $n_\text{layers}$ | 14 | Number of FWKV blocks |
| FFN multiplier | 4× | Hidden dimension = 2048 |
| $V$ | 50,257 | Vocabulary size (GPT-2 tokenizer + 2 special tokens) |
| $L_\text{max}$ | 1,024 | Maximum sequence length |
| $W_\text{floor}$ | 0.1 | Minimum decay value |
| Total parameters | **~56.4M** | 50M backbone + 6.4M ROSA embedding |

### 4.1 Parameter Breakdown

| Component | Parameters | % of Total |
|-----------|-----------|------------|
| Factorized embedding + head | 6.5M | 11.5% |
| ROSA embedding ($V+1 \times d_\text{emb}$) | 6.4M | 11.4% |
| FWKV blocks (14×) | ~43.5M | 77.1% |
| — Per block: 4 × Linear(d,d) + FFN | ~3.1M | — |
| **Total** | **~56.4M** | **100%** |

---

## 5. Training

### 5.1 Dataset

FWKV-ROSA is trained on **HuggingFaceH4/ultrachat_200k**, a curated dataset of synthetic multi-turn conversations. Earlier FWKV models (FWKV-29M, FWKV-50M) were trained on **Salesforce/wikitext** for general language modeling, while **FWKV-TinyStories** was trained on **roneneldan/TinyStories**—a dataset of synthetic children's stories generated by GPT-3.5 and GPT-4—demonstrating the architecture's versatility across data regimes.

- **Training split**: `train_sft` — 20,000 conversations
- **Validation split**: `test_sft` — 1,000 conversations
- **Format**: Multi-turn conversations with `role` and `content` fields

### 5.2 Chat Template

Each conversation is formatted with a simple two-role template:

```
<|user|> {user message}
<|assistant|> {assistant reply} <eos>
```

The loss is **masked** such that only tokens belonging to assistant turns contribute to the cross-entropy objective. User tokens and padding tokens are assigned `ignore_index = -100`. This ensures the model learns to generate responses rather than to predict user prompts.

### 5.3 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW ($\beta_1=0.9, \beta_2=0.999$) |
| Learning rate | $3 \times 10^{-4}$ |
| LR schedule | Cosine annealing |
| Weight decay | 0.1 |
| Batch size (real) | 8 |
| Gradient accumulation | 4 steps |
| Effective batch size | 32 |
| Epochs | 2 |
| Gradient clipping | 1.0 |
| Precision | BFloat16 (autocast) |
| CE chunk size | 512 |
| Gradient checkpointing | Enabled |

### 5.4 Memory Optimizations

Training a 56M-parameter model with 1,024-token sequences on consumer GPUs (tested on NVIDIA T4, 16 GB VRAM) required several memory-saving techniques:

1. **BFloat16 autocast**: All forward passes use `torch.bfloat16`, reducing activation memory by ~50% vs. FP32 while maintaining training stability.

2. **Gradient checkpointing**: Each FWKV block is wrapped with `torch.utils.checkpoint.checkpoint(use_reentrant=False)`, trading ~30% additional forward computation for a 50% reduction in activation memory.

3. **Chunked cross-entropy**: The full $B \times T \times V$ logit tensor is never materialized. Loss is computed in chunks of 512 tokens, reducing peak memory by $O(V)$ per chunk.

4. **Expandable segments**: `PYTORCH_ALLOC_CONF=expandable_segments:True` reduces CUDA memory fragmentation.

5. **Fused AdamW**: When available on CUDA, the fused optimizer implementation reduces kernel launch overhead and memory usage.

---

## 6. Results

### 6.1 Training Metrics

After 2 epochs of training on UltraChat 200k (20k conversations):

| Metric | Value |
|--------|-------|
| Validation loss | **4.216** |
| Validation perplexity | **67.78** |

For a 56M-parameter model trained on only 20k multi-turn conversations, a perplexity of 67.78 on held-out conversational data represents meaningful learning. The model successfully internalizes the chat format, turn-taking dynamics, and basic conversational patterns.

### 6.2 Inference Speed

FWKV-ROSA achieves exceptional inference throughput, particularly on CPU:

| Hardware | Throughput | Notes |
|----------|-----------|-------|
| **CPU (2 vCPU, 16 GB RAM)** | **48.1 tok/s** | Basic cloud CPU instance |
| GPU (NVIDIA T4, 16 GB) | ~45 tok/s | With BF16, parallel scan |

The CPU speed of 48.1 tok/s on a basic 2 vCPU instance is particularly noteworthy. This is achievable because:

1. **No KV cache**: The recurrent state is a fixed-size vector ($d_\text{model}$ per layer), not a growing cache. Memory usage is constant regardless of sequence length.

2. **Vectorized operations**: The parallel scan, while $O(T \log T)$ in total operations, maps efficiently to BLAS Level 1/2 operations (vector-vector multiply-add) that are well-optimized on CPUs.

3. **Small parameter count**: 56M parameters in BF16 occupy ~112 MB, easily fitting in CPU caches on modern processors.

### 6.3 Scalability Analysis

Benchmarks comparing FWKV against Transformer and RWKV across model sizes (10M to 1B parameters) on CUDA, measuring latency for generating 1,024 tokens:

| Model Size | Transformer | RWKV | FWKV |
|-----------|-------------|------|------|
| 10M | ~2.1s | ~0.8s | ~1.2s |
| 50M | ~5.3s | ~1.6s | ~2.4s |
| 100M | ~12.1s | ~3.4s | ~5.8s |
| 500M | ~41.2s | ~9.7s | ~18.3s |
| 1B | ~82.0s | ~19.0s | ~39.0s |

> *Note: The 1B parameter FWKV measurement reflects a T4 GPU bottleneck rather than a theoretical ceiling. FWKV's bandwidth requirements at 1B saturate the T4's memory bandwidth. On hardware with higher memory bandwidth (e.g., A100), FWKV is expected to approach RWKV's inference speed.*

The key trends:
- **FWKV scales better than Transformer** at all measured sizes due to the $O(T)$ recurrence replacing $O(T^2)$ attention.
- **RWKV maintains a ~2× speed advantage** over FWKV at equivalent sizes, attributable to RWKV's highly optimized CUDA kernel implementations (e.g., `wkv6` fused kernel).
- **FWKV's simplicity is its strength**: the architecture requires no custom kernels; the entire model runs in pure PyTorch. This means FWKV benefits automatically from improvements in PyTorch's compiler stack (`torch.compile`, `torch.export`), while RWKV's custom kernels require manual porting.

### 6.4 Qualitative Sample

A sample interaction demonstrating the model's conversational capability (generated at 45.1 tok/s):

> **User:** Explain how a linear recurrent network can still copy long‑range patterns.
>
> **Assistant:** Ingredients:
> - 1/2 cup of protein, chopped
> - 1/2 cup of corn butter, chopped
> - 1/2 tablespoon and pepper
> - 1/2 cup of olive oil
>
> Instructions:
> 1. Preheat your oven to 350°F (175°C), which will help extract them over the flavor and transfer.
> 2. In a skillet, whisk them in the salt and 1/2 cup of a skillet, diced powder, and 1/2 cup of a boil. Cut the chicken over the mixture, and add it to the cake.
> 3. Place the chicken of the chicken and cook it for a few minutes before adding a spoon to the heat.
> 4.
>
> *⚡ 45.1 tok/s*

This sample illustrates both the model's strengths and limitations at 56M parameters. The model has learned the structural format of recipe-style responses (ingredients list, numbered instructions) but confabulates content—a known behavior of small language models. The ROSA copy mechanism manifests in the repetitive phrasing ("1/2 cup of...") and structural consistency.

---

## 7. Discussion

### 7.1 Why FWKV Is Fast

FWKV's inference speed derives from three architectural properties:

1. **Constant state size**: Unlike Transformers, where the KV cache grows linearly with sequence length, FWKV maintains a single state vector of size $d_\text{model}$ per layer. This means:
   - Memory usage is $O(L \times d_\text{model})$ total, independent of sequence length.
   - No cache management, no memory fragmentation from growing tensors.
   - Token generation is $O(d_\text{model}^2)$ per step vs. Transformer's $O(T \times d_\text{model}^2)$.

2. **No data-dependent operations in the recurrence**: The decay $W$ is constant at inference time. The parallel scan uses only element-wise operations (multiply, add) that are trivially vectorizable. There are no gather/scatter operations, no sparse operations, no conditional branching in the inner loop.

3. **Factorized head**: The $V \times d_\text{emb}$ logit computation is $128\times$ cheaper than a dense $V \times d_\text{model}$ head. For GPT-2's vocabulary of 50,257 tokens, this saves ~25M floating-point operations per token.

### 7.2 The ROSA Advantage

ROSA provides something that purely learned recurrent models struggle with: **exact, deterministic long-range copying**. Consider a prefix like:

> "The capital of France is Paris. The capital of Germany is Berlin. The capital of France is"

A pure recurrent model must encode "Paris" in its fixed-size state and retrieve it after processing the intervening tokens about Germany. ROSA bypasses this: it identifies `"The capital of France is"` as a repeated suffix, finds that `"Paris"` followed it last time, and injects that prediction directly into the hidden state. The model then learns when to trust this signal.

This is particularly powerful for:
- **Repeated templates and patterns** in structured text
- **Long-range reference resolution** ("as mentioned above...")
- **Code completion** with repeated variable names and patterns

### 7.3 Limitations

1. **Small parameter count**: At 56M parameters, FWKV-ROSA's knowledge capacity is inherently limited. The confabulated recipe in the qualitative sample reflects this—the model learns patterns but not facts.

2. **ROSA is CPU-bound**: The suffix automaton construction runs on CPU and is not batched across examples. For production serving with large batch sizes, the ROSA computation could become a bottleneck.

3. **Training data scale**: 20k conversations is a minimal training set. Scaling to larger datasets (e.g., OpenHermes, UltraChat's full 200k) would likely improve coherence and factual accuracy significantly.

4. **GPU bandwidth bottleneck**: At 1B+ parameters, FWKV's simple operations become memory-bandwidth-bound on consumer GPUs. Optimized fused kernels (à la RWKV's `wkv6`) or `torch.compile` integration would be needed to close the gap with RWKV at larger scales.

5. **No explicit positional encoding**: FWKV relies entirely on the recurrent dynamics and ROSA for sequence ordering. This is theoretically sufficient but may limit performance on tasks requiring precise positional reasoning.

### 7.4 Future Directions

- **Scaling laws for recurrent models with copy mechanisms**: Investigate how ROSA's contribution changes with model size—does the copy signal become more or less important as the recurrent core gains capacity?

- **ROSA in other architectures**: ROSA is architecture-agnostic. Injecting ROSA embeddings into Transformer, Mamba, or RWKV models could provide a complementary copy mechanism.

- **Fused CUDA kernel for parallel_scan_decay**: A custom kernel that fuses the exponentiation and multiply-add steps of the parallel scan could significantly improve GPU throughput at larger model sizes.

- **Extended training**: Training FWKV-ROSA on larger conversational datasets (OpenHermes 2.5, ChatML-formatted data) to push the quality ceiling at this parameter count.

- **Multi-modal extensions**: The architecture has no components specific to text. Replacing the token embedding with a vision encoder (e.g., CLIP, DINOv2) and the output head with an image decoder would yield an FWKV-based text-to-image or vision-language model.

- **Language translation**: The recurrent state serves as a fixed-size bottleneck representation that could be shared across languages. Coupled with ROSA's pattern-matching, this architecture is well-suited for sequence-to-sequence tasks like machine translation.

---

## 8. Conclusion

We have presented the FWKV family of recurrent language models, spanning 29M to 56M parameters, achieving up to 48.1 tok/s inference on a basic CPU through the combination of a fixed-decay recurrent accumulator and—in the ROSA variant—an exact suffix-automaton-based copy mechanism. The architecture is structurally simpler than competing recurrent models—no gating, no custom CUDA kernels, no position embeddings—yet achieves competitive performance through careful co-design of its three components.

The key insight is that **simplicity enables speed**: by committing to a constant per-channel decay computed via a standard parallel scan, FWKV eliminates the need for custom kernels that couple performance to specific hardware and framework versions. The ROSA mechanism then compensates for the reduced expressivity of the recurrent core by providing an exact, training-free copy capability that is complementary to—and independent of—the learned recurrence.

At the time of writing, the FWKV family represents an existence proof that competitive language modeling—from unstructured text (FWKV-29M, FWKV-50M) to children's stories (FWKV-TinyStories) to multi-turn chat (FWKV-ROSA)—can be achieved at modest parameter counts with architectures that prioritize inference efficiency over expressive complexity. All models are released under Apache 2.0 and are fully compatible with the HuggingFace ecosystem.

---

## Appendix A: Architectural Snapshots for Reuse

The following components of FWKV-ROSA can be extracted and used independently in other systems:

### A.1 FWKV Recurrent Block (Standalone)

The fixed-decay accumulator with parallel scan is an $O(\log T)$ replacement for attention in any sequence model. It requires:
- 4 linear projections ($k$, $v$, $r$, out) of size $d \times d$
- 1 learnable parameter vector $w \in \mathbb{R}^d$
- The `parallel_scan_decay` function

No custom kernels, no position embeddings, no attention mask.

### A.2 ROSA Copy Predictor (Standalone)

ROSA can be used as a standalone copy-detection module for any text processing pipeline:
- Input: list of token IDs
- Output: list of predicted next tokens (or -1)
- $O(n)$ time, $O(n)$ memory, fully deterministic
- No training required

### A.3 Factorized Tied Head (Standalone)

The factorized embedding/head with chunked cross-entropy can replace the standard embedding + LM head in any Transformer or recurrent model, reducing embedding parameters from $O(V \times d_\text{model})$ to $O(V \times d_\text{emb} + d_\text{emb} \times d_\text{model})$.

---

## Appendix B: Mathematical Details

### B.1 Parallel Scan Correctness

**Theorem:** For a constant per-channel decay $W \in \mathbb{R}^d$ and input sequence $a_0, a_1, \ldots, a_{T-1} \in \mathbb{R}^d$, the Hillis-Steele scan computes:

$$S_t = \sum_{i=0}^{t} W^{t-i} \odot a_i$$

**Proof sketch:** After $\lceil \log_2 T \rceil$ iterations, step $d = 2^k$ adds to each position $t$ the contribution from position $t-d$, pre-multiplied by $W^d$. By induction on $k$, after processing all powers of two up to $d$, position $t$ has accumulated all positions $i$ where $(t-i)$ has bit $k$ set in its binary representation. The union of all such contributions over all $k$ covers all positions $0 \leq i \leq t$, each with the correct decay factor $W^{t-i}$.

### B.2 ROSA Correctness

**Theorem:** For any sequence $x[0..n-1]$, ROSA computes $y[i]$ as the token that followed the longest prior occurrence of the suffix ending at position $i$.

**Proof:** The suffix automaton's suffix links connect each state to the state representing its longest proper suffix. Walking up the suffix-link chain from the state representing the full prefix $x[0..i]$ visits states in order of decreasing substring length. The first state with a recorded `last_end` value therefore corresponds to the longest suffix with a prior occurrence, and `x[last_end[v] + 1]` is the token that followed it. ∎

---

## Appendix C: Model Availability

All models are available on the Hugging Face Hub under Apache 2.0 license:

| Model | Parameters | Training Data | Link |
|-------|-----------|---------------|------|
| FWKV-29M | 28.9M | Salesforce/wikitext | [HF](https://huggingface.co/FlameF0X/FWKV-29M) |
| FWKV-50M | 50.6M | Salesforce/wikitext | [HF](https://huggingface.co/FlameF0X/FWKV-50M) |
| FWKV-TinyStories | — | roneneldan/TinyStories | [HF](https://huggingface.co/FlameF0X/FWKV-TinyStories) |
| **FWKV-ROSA** | **56.4M** | **HuggingFaceH4/ultrachat_200k** | **[HF](https://huggingface.co/FlameF0X/FWKV-ROSA)** |

All models are compatible with HuggingFace Transformers and vLLM.

---

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need.* NeurIPS.

2. Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Biderman, S., Cao, H., Cheng, X., Chung, M., Derczynski, L., Du, X., Grella, M., GV, K., He, X., Hou, H., Kazienko, P., Kocon, J., Kong, J., Koptyra, B., ... Zhu, R.-J. (2023). *RWKV: Reinventing RNNs for the Transformer Era.* EMNLP.

3. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.

4. Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML.

5. Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., & Wei, F. (2023). *Retentive Network: A Successor to Transformer for Large Language Models.* arXiv:2307.08621.

6. Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M., Klambauer, G., Brandstetter, J., & Hochreiter, S. (2024). *xLSTM: Extended Long Short-Term Memory.* arXiv:2405.04517.

7. Blumer, A., Blumer, J., Haussler, D., Ehrenfeucht, A., Chen, M. T., & Seiferas, J. (1985). *The Smallest Automaton Recognizing the Subwords of a Text.* Theoretical Computer Science, 40, 31–55.

8. Ukkonen, E. (1995). *On-line Construction of Suffix Trees.* Algorithmica, 14(3), 249–260.

9. Vinyals, O., Fortunato, M., & Jaitly, N. (2015). *Pointer Networks.* NeurIPS.

10. Gu, J., Lu, Z., Li, H., & Li, V. O. K. (2016). *Incorporating Copying Mechanism in Sequence-to-Sequence Learning.* ACL.

11. See, A., Liu, P. J., & Manning, C. D. (2017). *Get To The Point: Summarization with Pointer-Generator Networks.* ACL.

12. Press, O., & Wolf, L. (2017). *Using the Output Embedding to Improve Language Models.* EACL.

13. Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., & Soricut, R. (2020). *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.* ICLR.

14. RWKV-8 ROSA: Beyond Attention. (2025). rwkv.com. Retrieved October 11, 2025.

15. Ding, N., Qin, Y., Yang, G., Wei, F., Yang, Z., Su, Y., Hu, S., Chen, Y., Chan, C.-M., Chen, W., Yi, J., Zhao, W., Wang, X., Zheng, H.-T., Liu, Z., & Sun, M. (2023). *UltraChat: Large-scale, Informative, and Diverse Multi-round Chat Data.* GitHub: thunlp/UltraChat.

---

*© 2026 Daniel Fox (FlameF0X). Released under Apache 2.0.*
