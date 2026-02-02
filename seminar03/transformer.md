<!-- markdownlint-disable MD033 -->

# II. Transformer

## 1. Vanilla Encoder–Decoder (Sutskever et al., 2014)

**Architecture:** RNN encoder $\to$ fixed vector $\to$ RNN decoder.  
No attention.

### Math

- Encoder reads inputs $ x_1, \dots, x_T $:

$$
h_t = \mathrm{RNN}_{\text{enc}}(h_{t-1}, x_t), \quad h_0 = 0
$$

- Final hidden state $ h_T $ is the **context vector** $ c $.

- Decoder generates outputs sequentially:

$$
s_t = \mathrm{RNN}_{\text{dec}}(s_{t-1}, y_{t-1}, c)
$$
$$
P(y_t \mid y_{<t}, x) = \mathrm{Softmax}(W_o s_t + b_o)
$$

## 2. Bahdanau (Additive) Attention (Bahdanau et al., 2014)

**Architecture:** RNN encoder–decoder with additive attention.  
Attention is computed **before** the decoder RNN step (_pre-RNN attention_).

## 3. Luong (Multiplicative) Attention (Luong et al., 2015)

**Architecture:** RNN encoder–decoder with multiplicative attention.  
Attention is computed **after** the decoder RNN step (_post-RNN attention_).  

## 4. Transformer Attention ([Attention Is All You Need [2017]](https://arxiv.org/abs/1706.03762))

**Architecture:** Encoder–decoder model without any RNN / LSTM / CNN / ..., do only attention.

So 2. and 3. did attention only _between_ encoder & decoder (that were RNN / ...), now encoder & decoder are attention themselves (and we also keep doing attention between).

### Step 1: Input Representations

$$
X \in \mathbb{R}^{n_{sequence} \times d_{\text{embedding}}}
$$
where row $X_i$ is the embedding of token $i$.  

### Step 2: $Q, K, V$

The Transformer learns **three linear projection matrices per head**:

$$
W^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad
W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad
W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}
$$

Then:

$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
$$

- $Q \in \mathbb{R}^{n \times d_k}$ (queries) - "What am I looking for?"

- $K \in \mathbb{R}^{n \times d_k}$ (keys) - "What properties do I have?"

- $V \in \mathbb{R}^{n \times d_v}$ (values) - "What information should I pass on if selected?"

### Step 3: Scoring Queries Against Keys

How much should token $i$ pay attention to token $j$?

The raw **attention score** between query $Q_i$ and key $K_j$ is:

$$
\text{score}(i,j) = Q_i \cdot K_j^T = \sum_{m=1}^{d_k} Q_{i,m} K_{j,m}
$$

$$
\text{Scores} = Q K^T \in \mathbb{R}^{n \times n}
$$

Row $i$ = how much token $i$ attends to all tokens.

### Step 4: Scaling by $\sqrt{d_k}$

#### **Assumptions**

1. $$
    \mathbb{E}[k_i] = \mathbb{E}[q_i] = 0, \quad \mathrm{Var}[k_i] = \mathrm{Var}[q_i] = 1
    $$

    - Under standard weight initialization (e.g., Xavier) that is okay

2. $ k_i, q_i $ are _independent_ across dimensions

Then:

$$
\mathrm{Var}[K^\top Q]
= \sum_{i=1}^{d_k} \mathrm{Var}[k_i q_i]
= \sum_{i=1}^{d_k} \mathrm{Var}[k_i] \mathrm{Var}[q_i]
= d_k
$$

#### **Scaling**

Because large values push softmax into saturation, leading to vanishing gradients, we scale by $\sqrt{d_k}$ (like temperature):

$$
\text{ScaledScores} = \frac{Q K^T}{\sqrt{d_k}}
$$

### Step 5: Attention

**Attention weights (scores)**:

$$
A = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times n}
$$

- Row $i$: distribution over which tokens $i$ attends to.  
- Each row sums to 1.  

**Attention output (pattern)**:

$$
\text{Attention}(Q, K, V) = AV
$$

### Step 6: Multi-Head Attention

> _multi_-head attention instead of a _single_ head to focus on different representation subspaces: Syntactic structure, Positional patterns, Semantic roles, ...

For $h$ heads, we repeat the above with different projection matrices:

$$
\text{head}_i = \text{Attention}(X W_i^Q, \, X W_i^K, \, X W_i^V)
$$

Then concatenate:

$$
\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

with $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$.

## Transformer Architecture Overview

<p align="center">
  <img src="transformer.png" alt="transformer" />
</p>

- **Encoder** (left block): processes the input sequence into a contextual representation.

- **Decoder** (right block): generates the output sequence one token at a time.

  Attends both to previously generated tokens and the encoder’s representation:

  - Queries $\{Q_i\}$ come from the decoder.
  - Keys $\{K_i\}$ & Values $\{V_i\}$ come from encoder output.

  Allows decoder to focus on relevant input parts when generating output.

> Embeddings flow from the last layer of Encoder block into all Decoder block's attention (not from the respective (соответствующих) layers).

### Masked Multi-Head Attention

The mask is just 0&1 matrix for decoder not to look into future.

## BERT-GPT-BART

1. BERT (encoder-only)

    Bidirectional Encoder Representations from Transformers.

2. GPT-x (decoder-only)

    Generative Pretrained Transformer.

3. BART (encoder-decoder)

    Encoder reads corrupted text in full (like BERT).

    Decoder generates clean text autoregressively (like GPT).

### Why GPT-x, but not BART?
