> *Author: Daniel Fox (FlameF0X)*
> *Hugging Face: [FlameF0X](https://huggingface.co/FlameF0X), [FWKV](https://huggingface.co/FWKV)*
> *Project: [FWKV](https://huggingface.co/FWKV)*
> *Date: October 2026*

---

## Abstract

We describe the architecture of FWKV, a recurrent language model that replaces quadratic self-attention with a stack of per-channel linear recurrences. Each channel carries a single scalar memory whose decay is fixed at inference time, and the recurrence is evaluated in training as a causal fast Fourier transform (FFT) convolution in fp32. The block design combines pre-normalization, RWKV-style token shift, a unit-variance input gain, a per-head output normalization, and a receptance gate. Decay rates are initialized across timescales from 2 to 1000 tokens, so that different channels begin with different memory horizons. The output head is factorized and tied to the input embedding, and the training loss is computed in checkpointed chunks. We give the full forward equations, derive the equivalence between the convolutional training form and the recurrent inference form, analyze time and memory complexity, and describe the invariants we use to verify the implementation. The architecture requires no custom kernels and no positional encodings, and its per-token inference state is a fixed-size tensor for each layer.

---

## 1. Introduction

Transformers [Vaswani et al., 2017] compute each output as a weighted sum over all previous positions. This costs $O(T^2)$ time in training and yields a key-value cache that grows linearly with the sequence length during generation. A family of recurrent alternatives replaces this with a state of fixed size, including RWKV [Peng et al., 2023], linear attention [Katharopoulos et al., 2020], retentive networks [Sun et al., 2023], and selective state-space models [Gu & Dao, 2023].

FWKV occupies a narrow point in this space. Its recurrence is linear in the state and has a decay that is a learned constant per channel, not a function of the input. Because the decay does not depend on the data, the recurrence is time-invariant, and it can be evaluated either as a convolution over the whole sequence (training) or as a step update (inference). The main contributions of this paper are:

1. A complete specification of the FWKV block, including token shift, decay parameterization, unit-gain input normalization, and per-head output normalization.
2. A multi-timescale initialization of the decay that assigns each channel a memory horizon log-uniformly between 2 and 1000 tokens.
3. An FFT-based evaluation of the recurrence in fp32, with a proof that it equals the recurrent form.
4. A set of architectural invariants and tests that guarantee cached decoding matches full-sequence evaluation.

This paper concerns the architecture only. It does not report training curves, benchmark scores, or the behavior of any particular checkpoint.

---

## 2. Background

### 2.1 Linear Recurrences

A first-order linear recurrence over a sequence $a_0, \ldots, a_{T-1}$ with a diagonal decay $W \in (0,1)^d$ is

$$y_t = a_t + W \odot y_{t-1}, \qquad y_{-1} = s,$$

where $s$ is an optional carried-in state. Unrolling gives

$$y_t = W^{t+1} \odot s + \sum_{j=0}^{t} W^{t-j} \odot a_j.$$

The sum is a causal convolution of $a$ with the kernel $\kappa_j = W^j$. This observation is the basis for both the training and the inference forms in this paper.

### 2.2 Token Shift

RWKV [Peng et al., 2023] mixes each position's input with its predecessor before the projections. This gives every block a direct, cheap view of the previous token without an attention window. FWKV adopts this idea. Given normalized inputs $h_t$ and per-channel mixing coefficients $\mu \in [0,1]^d$,

$$\tilde{h}_t = \mu \odot h_t + (1-\mu) \odot h_{t-1}.$$

At $t=0$, $h_{-1}$ is the carried-in shift state, which is zero when no state is provided.

---

## 3. Architecture

### 3.1 Overview

An FWKV model is a stack of $L$ identical-form blocks between a factorized embedding and a tied output head. Figure 1 shows the top-level data path.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 560" width="640" height="560" font-family="Helvetica, Arial, sans-serif" role="img" aria-label="FWKV top-level architecture">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#444"/>
    </marker>
  </defs>
  <rect x="200" y="10" width="240" height="44" rx="6" fill="#f2f2f2" stroke="#444"/>
  <text x="320" y="37" text-anchor="middle" font-size="14">Token ids  [B, T]</text>

  <line x1="320" y1="54" x2="320" y2="82" stroke="#444" stroke-width="1.5" marker-end="url(#arr)"/>

  <rect x="150" y="84" width="340" height="54" rx="6" fill="#dbe8f7" stroke="#2d5f8b"/>
  <text x="320" y="107" text-anchor="middle" font-size="14">Factorized embedding</text>
  <text x="320" y="125" text-anchor="middle" font-size="12">E[x] in R^{d_emb},  then  proj to R^{d}</text>

  <line x1="320" y1="138" x2="320" y2="166" stroke="#444" stroke-width="1.5" marker-end="url(#arr)"/>

  <rect x="200" y="168" width="240" height="36" rx="6" fill="#f2f2f2" stroke="#444"/>
  <text x="320" y="191" text-anchor="middle" font-size="14">LayerNorm (ln0)</text>

  <line x1="320" y1="204" x2="320" y2="232" stroke="#444" stroke-width="1.5" marker-end="url(#arr)"/>

  <rect x="90" y="234" width="460" height="176" rx="8" fill="#fff" stroke="#2d5f8b" stroke-dasharray="5 4"/>
  <text x="102" y="254" font-size="12" fill="#2d5f8b">FWKV block  x L</text>
  <rect x="150" y="266" width="340" height="60" rx="6" fill="#e8f3e4" stroke="#3d7a2f"/>
  <text x="320" y="290" text-anchor="middle" font-size="14">Time mixing (WKV)</text>
  <text x="320" y="310" text-anchor="middle" font-size="12">token shift, decay scan, receptance gate</text>
  <line x1="320" y1="326" x2="320" y2="342" stroke="#444" stroke-width="1.5" marker-end="url(#arr)"/>
  <rect x="150" y="344" width="340" height="50" rx="6" fill="#fbeedd" stroke="#9a5b13"/>
  <text x="320" y="366" text-anchor="middle" font-size="14">Channel mixing (FFN)</text>
  <text x="320" y="384" text-anchor="middle" font-size="12">token shift, GELU MLP, 4x expansion</text>

  <line x1="320" y1="410" x2="320" y2="436" stroke="#444" stroke-width="1.5" marker-end="url(#arr)"/>

  <rect x="200" y="438" width="240" height="36" rx="6" fill="#f2f2f2" stroke="#444"/>
  <text x="320" y="461" text-anchor="middle" font-size="14">LayerNorm (norm)</text>

  <line x1="320" y1="474" x2="320" y2="498" stroke="#444" stroke-width="1.5" marker-end="url(#arr)"/>

  <rect x="110" y="500" width="420" height="44" rx="6" fill="#f2f2f2" stroke="#444"/>
  <text x="320" y="527" text-anchor="middle" font-size="13">proj^T to R^{d_emb}, then tied logits  E^T  (or chunked CE)</text>
</svg>

**Figure 1.** Top-level FWKV data path. The embedding and output head share the matrix $E \in \mathbb{R}^{V \times d_\text{emb}}$. Each block contains a time-mixing sub-layer and a channel-mixing sub-layer, both pre-normalized and residual.

### 3.2 Factorized Tied Embedding and Head

Let $V$ be the vocabulary size, $d$ the model width, and $d_\text{emb}$ the embedding width. The model stores one matrix $E \in \mathbb{R}^{V \times d_\text{emb}}$ and one projection $P \in \mathbb{R}^{d \times d_\text{emb}}$ (no bias). The input embedding is

$$x_t^{(0)} = \text{LN}_0\big(P\, E[u_t]\big),$$

where $u_t$ is the token id. The output projects the final hidden state back into embedding space with the transpose of $P$ and scores it against $E$:

$$\hat{z}_t = E\, P^{\top} \text{LN}_f(x_t^{(L)}) \in \mathbb{R}^{V}.$$

Note that $P^{\top}$ is a transposed projection, not an exact inverse, because $P$ is not square when $d \neq d_\text{emb}$. The model learns to make this path useful. The parameter cost of the embedding is $V d_\text{emb} + d\, d_\text{emb}$ instead of $V d$. With $V = 50{,}257$, $d_\text{emb} = 384$, and $d = 768$, this is about $19.6\text{M}$ parameters instead of about $38.6\text{M}$.

The loss is computed by chunked cross-entropy over the rows of $E$. Each chunk's logits are recomputed during backward, so the full $(B \cdot T) \times V$ logit matrix is never held in memory.

### 3.3 Pre-Normalized Residual Blocks

Each block $\ell$ receives $x^{(\ell)} \in \mathbb{R}^{B \times T \times d}$ and returns $x^{(\ell+1)}$ through two residual updates:

$$x \leftarrow x + \text{TimeMix}_\ell(\text{LN}_1(x)), \qquad x \leftarrow x + \text{ChanMix}_\ell(\text{LN}_2(x)).$$

Pre-normalization keeps the residual stream on a stable scale across depth. Output projections of both sub-layers are initialized with a smaller standard deviation (Section 4.1), so each block starts close to the identity.

### 3.4 Token Shift

Let $h = \text{LN}_1(x)$ and let $h_{\text{prev}}$ be $h$ shifted by one position, with $h_{\text{prev},0}$ taken from the carried-in shift state. Three mixed inputs are formed for keys, values, and receptance:

$$\begin{aligned}
x_k &= \mu_k \odot h + (1-\mu_k) \odot h_{\text{prev}},\\
x_v &= \mu_v \odot h + (1-\mu_v) \odot h_{\text{prev}},\\
x_r &= \mu_r \odot h + (1-\mu_r) \odot h_{\text{prev}}.
\end{aligned}$$

Each $\mu$ is a learned vector in $\mathbb{R}^d$ with its own initialization (Section 4.2). The channel-mixing sub-layer applies the same operation with its own vector $\mu_c$ and its own shift state.

### 3.5 Decay Parameterization

Each channel $c$ has a learned logit $w_c$. The decay is confined to a fixed interval:

$$W_c = \alpha + (\beta - \alpha)\,\sigma(w_c), \qquad 0 < \alpha < \beta < 1,$$

with $\alpha = 0.1$ and $\beta = 0.999$ by default. Because the sigmoid is smooth everywhere, the gradient with respect to $w_c$ never vanishes by clamping. The value $W_c$ cannot reach $\beta$, which bounds the longest memory a channel can hold.

A channel with decay $W$ has an effective memory horizon $\tau = 1/(1-W)$, the number of steps over which its kernel decays by a factor of $e$. The floor $\alpha$ bounds how quickly a channel can forget.

### 3.6 Unit-Gain Input and the Scan

The time-mixing sub-layer forms per-channel inputs

$$a_t = \sqrt{1 - W^2} \odot \big(k_t \odot v_t\big).$$

If $k_t$ and $v_t$ have unit variance and are independent over time, the stationary variance of $y_t$ defined below is approximately one for every channel, independent of $W$. Without this gain, channels with $W$ near $1$ would accumulate variance proportional to $\tau$, and the scale of the mixer output would depend on the timescale assigned to the channel.

The scan is

$$y_t = a_t + W \odot y_{t-1}, \qquad y_{-1} = s,$$

which equals the convolution from Section 2.1. In training, the whole sequence is evaluated at once. Let $n = 2^{\lceil \log_2 (2T-1) \rceil}$, so that linear convolution of two length-$T$ signals fits in $n$ points without circular wrap-around. Let $K_{t,c} = W_c^t$ for $t = 0, \ldots, T-1$. Then

$$Y = \text{irfft}_n\!\big(\text{rfft}_n(A) \odot \text{rfft}_n(K)\big)_{[0:T]}.$$

The carried-in state is folded into the first input as $a_0 \leftarrow a_0 + W \odot s$ before the convolution, which is exact because $s$ enters the recurrence only through $y_{-1}$. All of this runs in fp32, with autocast disabled, because the FFT and the powers $W^t$ lose precision in half precision at long lengths.

The cost is $O(d\, T \log T)$ per block. The file retains a Hillis-Steele log-depth scan as a reference implementation for testing. It is not used in the model, since the FFT path is faster on the sequence lengths used here and gives the same result (Section 7).

### 3.7 Per-Head Output Normalization and Receptance Gate

The scan output $y$ is normalized by a GroupNorm with $d / d_h$ groups, where $d_h$ is the head size (64 by default, giving 12 groups at $d = 768$). Each head is normalized independently, which keeps the scale of each head stable even when some heads have very long memories. The gate is then applied:

$$\text{WKV}_t = r_t \odot \text{GN}(y_t), \qquad r_t = \sigma\big(W_r x_r\big),$$

$$x \leftarrow x + W_o\, \text{WKV}_t.$$

The receptance gate $r_t$ is a learned, data-dependent sigmoid. It is the only data-dependent multiplicative element in the time-mixing path, and it is the mechanism by which the model selects which channels to pass into the residual stream at each position.

### 3.8 Channel Mixing

The channel-mixing sub-layer uses the same token shift and a two-layer MLP with GELU activation:

$$x \leftarrow x + W_2\, \text{GELU}\big(W_1\, (\mu_c \odot h_2 + (1-\mu_c) \odot h_{2,\text{prev}})\big),$$

with $W_1 \in \mathbb{R}^{4d \times d}$ and $W_2 \in \mathbb{R}^{d \times 4d}$, both without bias. Figure 2 shows one block.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 600" width="640" height="600" font-family="Helvetica, Arial, sans-serif" role="img" aria-label="FWKV block dataflow">
  <defs>
    <marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#444"/>
    </marker>
  </defs>
  <text x="20" y="24" font-size="13" font-weight="bold">Time mixing</text>
  <rect x="20" y="34" width="600" height="230" rx="8" fill="none" stroke="#3d7a2f" stroke-dasharray="5 4"/>

  <rect x="40" y="60" width="90" height="40" rx="5" fill="#f2f2f2" stroke="#444"/>
  <text x="85" y="85" text-anchor="middle" font-size="12">LN_1</text>
  <line x1="130" y1="80" x2="160" y2="80" stroke="#444" marker-end="url(#arr2)"/>

  <rect x="160" y="60" width="130" height="40" rx="5" fill="#e8f3e4" stroke="#3d7a2f"/>
  <text x="225" y="78" text-anchor="middle" font-size="12">token shift</text>
  <text x="225" y="92" text-anchor="middle" font-size="11">mu_k, mu_v, mu_r</text>
  <line x1="290" y1="80" x2="320" y2="80" stroke="#444" marker-end="url(#arr2)"/>

  <rect x="320" y="60" width="90" height="40" rx="5" fill="#fff" stroke="#444"/>
  <text x="365" y="78" text-anchor="middle" font-size="12">W_k, W_v</text>
  <text x="365" y="92" text-anchor="middle" font-size="11">k, v</text>
  <line x1="410" y1="80" x2="440" y2="80" stroke="#444" marker-end="url(#arr2)"/>

  <rect x="440" y="60" width="160" height="40" rx="5" fill="#dbe8f7" stroke="#2d5f8b"/>
  <text x="520" y="78" text-anchor="middle" font-size="12">a = sqrt(1-W^2) k v</text>
  <text x="520" y="92" text-anchor="middle" font-size="11">unit-gain input</text>

  <line x1="520" y1="100" x2="520" y2="130" stroke="#444" marker-end="url(#arr2)"/>

  <rect x="400" y="132" width="240" height="46" rx="5" fill="#dbe8f7" stroke="#2d5f8b"/>
  <text x="520" y="152" text-anchor="middle" font-size="12">decay scan  y_t = a_t + W y_(t-1)</text>
  <text x="520" y="169" text-anchor="middle" font-size="11">fp32 FFT conv / state carry</text>

  <line x1="400" y1="155" x2="350" y2="155" stroke="#444" marker-end="url(#arr2)"/>

  <rect x="200" y="132" width="150" height="46" rx="5" fill="#f2f2f2" stroke="#444"/>
  <text x="275" y="152" text-anchor="middle" font-size="12">GroupNorm (per head)</text>
  <text x="275" y="169" text-anchor="middle" font-size="11">d / d_h groups</text>

  <line x1="200" y1="155" x2="160" y2="155" stroke="#444" marker-end="url(#arr2)"/>

  <rect x="40" y="132" width="110" height="46" rx="5" fill="#fff" stroke="#444"/>
  <text x="95" y="152" text-anchor="middle" font-size="12">r = sigma(W_r x_r)</text>
  <text x="95" y="169" text-anchor="middle" font-size="11">receptance</text>

  <line x1="95" y1="178" x2="95" y2="216" stroke="#444" marker-end="url(#arr2)"/>
  <line x1="200" y1="155" x2="130" y2="216" stroke="#444" marker-end="url(#arr2)"/>

  <circle cx="110" cy="230" r="11" fill="#fff" stroke="#444"/>
  <text x="110" y="234" text-anchor="middle" font-size="13">x</text>
  <line x1="121" y1="230" x2="150" y2="230" stroke="#444" marker-end="url(#arr2)"/>

  <rect x="150" y="212" width="140" height="36" rx="5" fill="#fff" stroke="#444"/>
  <text x="220" y="234" text-anchor="middle" font-size="12">W_o, then + residual</text>

  <text x="20" y="300" font-size="13" font-weight="bold">Channel mixing</text>
  <rect x="20" y="310" width="600" height="150" rx="8" fill="none" stroke="#9a5b13" stroke-dasharray="5 4"/>

  <rect x="40" y="336" width="90" height="40" rx="5" fill="#f2f2f2" stroke="#444"/>
  <text x="85" y="361" text-anchor="middle" font-size="12">LN_2</text>
  <line x1="130" y1="356" x2="160" y2="356" stroke="#444" marker-end="url(#arr2)"/>

  <rect x="160" y="336" width="130" height="40" rx="5" fill="#fbeedd" stroke="#9a5b13"/>
  <text x="225" y="354" text-anchor="middle" font-size="12">token shift</text>
  <text x="225" y="368" text-anchor="middle" font-size="11">mu_c</text>
  <line x1="290" y1="356" x2="320" y2="356" stroke="#444" marker-end="url(#arr2)"/>

  <rect x="320" y="336" width="130" height="40" rx="5" fill="#fff" stroke="#444"/>
  <text x="385" y="354" text-anchor="middle" font-size="12">W_1 , GELU , W_2</text>
  <text x="385" y="368" text-anchor="middle" font-size="11">4d hidden, no bias</text>
  <line x1="450" y1="356" x2="480" y2="356" stroke="#444" marker-end="url(#arr2)"/>

  <rect x="480" y="336" width="130" height="40" rx="5" fill="#fff" stroke="#444"/>
  <text x="545" y="361" text-anchor="middle" font-size="12">+ residual</text>

  <text x="20" y="530" font-size="12" fill="#444">Carried state per layer: (WKV state in R^d, time-mix shift in R^d, channel-mix shift in R^d)</text>
  <text x="20" y="550" font-size="12" fill="#444">Each block's recurrence is time-invariant; the only per-step cost is O(d^2) for the projections.</text>
</svg>

**Figure 2.** One FWKV block. The time-mixing sub-layer contains the only recurrence. The channel-mixing sub-layer is a token-shifted MLP. Each layer carries three state tensors, each of size $d$, and nothing else.

### 3.9 Parameter Count per Block

Per block, the time-mixing sub-layer has four $d \times d$ matrices ($W_k, W_v, W_r, W_o$), and the channel-mixing sub-layer has two matrices of size $4d \times d$. The block also has $d$ decay logits, four token-shift vectors, and two pairs of LayerNorms with GroupNorm affine parameters. The total is

$$P_\text{block} = 12 d^2 + O(d).$$

At $d = 768$ this is about $7.1\text{M}$ parameters per block.

---

## 4. Initialization and Normalization

### 4.1 Weight Initialization

Linear weights are drawn from $\mathcal{N}(0, 0.02^2)$. Output projections of the two residual sub-layers ($W_o$ and the second FFN matrix) are drawn with standard deviation $0.02 / \sqrt{2L}$, where $L$ is the number of blocks. This keeps the variance of the residual stream roughly constant with depth. Biases are zero where they exist. LayerNorm and GroupNorm scales are one and offsets are zero.

### 4.2 Token-Shift Coefficients

For block $\ell$ out of $L$, define $\rho_1 = 1 - \ell/L$, $\rho_2 = \ell / (L-1)$, and a ramp $\text{ramp}_c = c/d$. The coefficients are

$$\mu_k = \text{ramp}^{\rho_1}, \quad \mu_v = \text{ramp}^{\rho_1} + 0.3\,\rho_2, \quad \mu_r = \text{ramp}^{\rho_1/2}, \quad \mu_c = \text{ramp}^{\rho_1}.$$

Early blocks start with coefficients near one for most channels, which means they mostly use the current token. Later blocks start with small coefficients, which means they draw more from the previous token. This follows the RWKV initialization pattern [Peng et al., 2023].

### 4.3 Multi-Timescale Decay Initialization

Each channel is assigned a target memory horizon on a logarithmic grid:

$$\tau_c = \exp\!\Big(\ln \tau_\text{min} + \frac{c}{d-1}\big(\ln \tau_\text{max} - \ln \tau_\text{min}\big)\Big), \qquad \tau_\text{min} = 2,\; \tau_\text{max} = 1000.$$

The corresponding decay is $W_c^{(0)} = 1 - 1/\tau_c$, clamped into the open interval $(\alpha, \beta)$. The logit is obtained by inverting the parameterization of Section 3.5:

$$w_c = \ln\!\frac{p_c}{1-p_c}, \qquad p_c = \frac{W_c^{(0)} - \alpha}{\beta - \alpha}.$$

Figure 3 shows the resulting horizon as a function of channel index. Every layer begins with the same spread of horizons, from about 2 tokens to about 1000 tokens. Training then moves each channel's decay as needed. The initialization guarantees that the model is not forced to begin as a short-memory system, which was a limitation of earlier versions that used a single constant decay for all channels.

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 300" width="640" height="300" font-family="Helvetica, Arial, sans-serif" role="img" aria-label="Initial memory horizon by channel">
  <line x1="60" y1="30" x2="60" y2="230" stroke="#444"/>
  <line x1="60" y1="230" x2="580" y2="230" stroke="#444"/>
  <text x="56" y="234" text-anchor="end" font-size="11">2</text>
  <text x="56" y="184" text-anchor="end" font-size="11">10</text>
  <text x="56" y="134" text-anchor="end" font-size="11">100</text>
  <text x="56" y="84" text-anchor="end" font-size="11">500</text>
  <text x="56" y="34" text-anchor="end" font-size="11">1000</text>
  <text x="60" y="252" text-anchor="middle" font-size="11">0</text>
  <text x="300" y="252" text-anchor="middle" font-size="11">channel index (fraction of d)</text>
  <text x="580" y="252" text-anchor="middle" font-size="11">1</text>
  <text x="18" y="130" font-size="11" transform="rotate(-90 18 130)" text-anchor="middle">memory horizon tau (tokens, log scale)</text>
  <polyline points="60,230 180,180 300,130 420,80 540,30" fill="none" stroke="#2d5f8b" stroke-width="2.5"/>
  <circle cx="60" cy="230" r="4" fill="#2d5f8b"/>
  <circle cx="180" cy="180" r="4" fill="#2d5f8b"/>
  <circle cx="300" cy="130" r="4" fill="#2d5f8b"/>
  <circle cx="420" cy="80" r="4" fill="#2d5f8b"/>
  <circle cx="540" cy="30" r="4" fill="#2d5f8b"/>
  <text x="66" y="222" font-size="11">2</text>
  <text x="186" y="172" font-size="11">9.5</text>
  <text x="306" y="122" font-size="11">45</text>
  <text x="426" y="72" font-size="11">211</text>
  <text x="446" y="22" font-size="11">1000</text>
</svg>

**Figure 3.** Initial memory horizon $\tau_c$ across channels at a single layer, sampled at five points on the log grid. Each layer starts with the same spread. The decay $W_c = 1 - 1/\tau_c$ is then learned.

### 4.4 Summary of Stability Mechanisms

Five mechanisms keep the recurrence and the residual stream stable: the bounded sigmoid decay (no clamp, so no dead gradient), the unit-gain input scaling, pre-normalization, per-head output normalization, and the scaled output initialization. None of these depends on a specific data distribution.

---

## 5. Inference and State

### 5.1 Carried State

Each block carries three tensors of shape $(B, d)$:

1. The WKV state $y_{T-1}$ from the scan.
2. The time-mix shift state, which is the last normalized input $h_{T-1}$.
3. The channel-mix shift state, which is the last normalized input $h^{(2)}_{T-1}$.

The total state size is $3Ld$ values per sequence, independent of the number of tokens processed. The model has no key-value cache and no position index.

### 5.2 Prefill and Step Decoding

Prefill processes the prompt in one pass using the FFT path, which returns the final state of each block. Each generated token is then processed by a single-step update: the token shift uses the stored previous input, the scan reduces to $y_t = a_t + W \odot y_{t-1}$, and the channel mixer runs on one vector. The forward function accepts the carried state through a single argument so that the same code path serves training (no state) and generation (with state).

### 5.3 Equivalence of the Two Forms

**Claim.** For any input sequence, the outputs of step decoding, starting from the state produced by prefill, equal the outputs of a single full-sequence forward pass.

**Argument.** Token shift depends only on the previous normalized input, which is stored in the shift state. The scan is linear and time-invariant, so the FFT convolution with the state folded into $a_0$ and the sequential recurrence produce the same $y_t$ up to floating-point error. GroupNorm and the gate are pointwise in time. The channel mixer is pointwise in time apart from its shift. Hence every operation is exact across the boundary between prefill and step decoding.

The file includes a test that checks this claim numerically. It runs a full forward pass on a random sequence, then compares it with a prefill of the first 12 tokens followed by 8 single-token steps. The maximum absolute logit difference must be below $10^{-3}$. This test would catch errors in state handling, token-shift indexing, and the scan.

---

## 6. Complexity

Let $T$ be the sequence length, $d$ the width, $L$ the depth, and $V$ the vocabulary size.

| Quantity | Training (per sequence) | Generation (per token) |
|---|---|---|
| Projections (time and channel mixing) | $O(T L d^2)$ | $O(L d^2)$ |
| Decay scan | $O(T L d \log T)$ (FFT) | $O(L d)$ |
| Output head | $O(T d_\text{emb} V + T d\, d_\text{emb})$ | $O(d_\text{emb} V + d\, d_\text{emb})$ |
| Recurrent state memory | $O(T L d)$ (activations) | $O(L d)$ |

The scan's $\log T$ factor is the cost of the FFT, which is present only in training and only on the sequence lengths used in a batch. Generation avoids the FFT entirely. Compared with attention, the per-token generation cost does not grow with the length of the context, and the state memory does not grow either.

The output head dominates the cost of generation at large vocabularies. The factorized head reduces its cost from $O(dV)$ to $O(d_\text{emb} V + d\, d_\text{emb})$, a factor of $d/d_\text{emb}$ on the dominant term.

---

## 7. Implementation Notes and Verification

### 7.1 Numerical Precision

The scan, the decay powers $W^t = \exp(t \ln W)$, and the FFT run in fp32 regardless of the autocast setting. The rest of the block runs under autocast in fp16 on pre-Ampere GPUs and in bf16 on newer GPUs. Mixed-precision training uses a gradient scaler in fp16 mode. The logit scaling and loss are computed in fp32 inside the checkpointed chunk function.

### 7.2 Gradient Checkpointing of the Head

The chunked cross-entropy wraps each chunk in non-reentrant checkpointing when gradients are required. This recomputes the chunk logits during backward. The effect is that peak memory for the head depends on the chunk size, not on the number of tokens in the batch.

### 7.3 Invariants Checked by Tests

The reference implementation includes three checks, run on CPU in seconds:

1. The FFT scan agrees with the Hillis-Steele reference scan, with and without carried state.
2. Cached step decoding agrees with the full forward pass (Section 5.3).
3. The loss is finite at initialization, and gradients reach the decay logits, the token-shift coefficients, and the projection matrices of the first block.

### 7.4 Forward-Pass Structure

The model's forward method enters the autocast context inside the function itself, rather than around the caller. This matters for data-parallel replication, because autocast state is thread-local and must be set on each replica. The forward method also returns either logits or a loss, depending on whether labels are provided, so the same module serves training, evaluation, and generation.

---

## 8. Discussion

### 8.1 Why Fixed Decay

The decay $W_c$ does not depend on the input. This is a deliberate restriction. It makes the recurrence a convolution, which can be computed with standard FFT primitives and does not need a custom associative-scan kernel. The price is that the model cannot forget a specific token on demand. Forgetting is controlled instead by the receptance gate, which is data-dependent, and by the choice of channel. A channel with a short horizon forgets quickly regardless of content.

### 8.2 Why Multi-Timescale Initialization

Earlier versions gave every channel the same decay, which caused every layer to have roughly the same memory horizon of about eight tokens. Such a model can only use context within a short window unless training changes the decays substantially, and the gradients for the decay logits are weak. The logarithmic initialization assigns each layer a range of horizons from the first step, so local and long-range structure are both representable from the start.

### 8.3 Why Unit-Gain Input

Without the $\sqrt{1-W^2}$ factor, the variance of the scan output for a channel with decay $W$ grows roughly as $1/(1-W^2)$. For channels with horizons near 1000 tokens, this would be about five hundred times the variance of short-horizon channels. The gain removes this dependence, so that the scale of the mixer output is set by the projections rather than by the timescale.

### 8.4 Limitations

1. **Fixed decay limits selectivity.** A channel's forgetting rate does not respond to content. Architectures with input-dependent transitions can express some patterns that FWKV can only approximate through the gate.
2. **Factorized head is approximate.** The output path uses the transpose of a rectangular projection rather than an inverse, so the logit computation is a learned bottleneck, not an exact factorization.
3. **FFT training cost has a $\log T$ factor and an fp32 requirement.** The fp32 scan uses more memory and bandwidth than a fused half-precision kernel would. A dedicated kernel could close this gap.
4. **No positional encoding.** Order is carried entirely by the recurrence and token shift. This may limit precise positional reasoning, and we have not characterized this limit here.

---

## 9. Conclusion

FWKV is a recurrent language model whose architecture is defined by four choices: a fixed, per-channel, input-independent decay with a multi-timescale initialization; a token-shifted time-mixing sub-layer whose recurrence is evaluated as an FFT convolution in training and as a single-step update in generation; unit-gain inputs and per-head output normalization that keep the mixer's scale independent of the decay; and a factorized, tied embedding and head with checkpointed loss. Together these give a model with a fixed-size state per layer, no attention, no positional table, and no custom kernels. The main open questions concern how much of the expressivity lost by fixed decay is recovered by the receptance gate and depth, and how the architecture scales on hardware with higher memory bandwidth.

---

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need.* NeurIPS.

2. Peng, B., et al. (2023). *RWKV: Reinventing RNNs for the Transformer Era.* EMNLP.

3. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.

4. Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML.

5. Sun, Y., et al. (2023). *Retentive Network: A Successor to Transformer for Large Language Models.* arXiv:2307.08621.

6. Hillis, W. D., & Steele, G. L. (1986). *Data Parallel Algorithms.* Communications of the ACM, 29(12), 1170–1183.

7. Cooley, J. W., & Tukey, J. W. (1965). *An Algorithm for the Machine Calculation of Complex Fourier Series.* Mathematics of Computation, 19(90), 297–301.

8. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization.* arXiv:1607.06450.

9. Wu, Y., & He, K. (2018). *Group Normalization.* ECCV.

10. Hendrycks, D., & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs).* arXiv:1606.08415.

11. Press, O., & Wolf, L. (2017). *Using the Output Embedding to Improve Language Models.* EACL.

12. Lan, Z., et al. (2020). *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.* ICLR.

13. Chen, T., Xu, B., Zhang, C., & Guestrin, C. (2016). *Training Deep Nets with Sublinear Memory Cost.* arXiv:1604.06174.

14. Micikevicius, P., et al. (2018). *Mixed Precision Training.* ICLR.

15. Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR.

---

*© 2026 Daniel B. (FlameF0X). Released under Apache 2.0.*
