# LLaMA PyTorch Implementation

This repository contains a clean, modular, and optimized from-scratch PyTorch implementation of the **LLaMA** (specifically LLaMA 2) large language model architecture. 

It is tailored for educational purposes, inference experimentation, and examining the micro-mechanics of modern transformer models.

---

## Architecture Overview

Unlike standard Vanilla Transformers, the LLaMA architecture introduces several optimizations for efficiency, stability, and scaling:
1. **RMSNorm (Root Mean Square Normalization)**: Applied before attention and feed-forward layers.
2. **Rotary Position Embeddings (RoPE)**: Replaces absolute or relative positional embeddings with a frequency-based rotation in complex space.
3. **Grouped-Query Attention (GQA)**: Accelerates inference scale using fewer Key/Value heads than Query heads, reducing KV cache memory footprint.
4. **SwiGLU Activation FFN**: Uses gated linear units (GLUs) with SiLU (Swish) activations inside feedforward sub-layers.

Here is the computational graph of a single **Transformer Decoder Block** in LLaMA:

```mermaid
graph TD
    Input["Input Tensor: x <br>Shape: B, Seq_Len, Dim"] --> RMSNorm1[RMSNorm]
    
    subgraph self_attn ["Self-Attention Block"]
        RMSNorm1 --> WQ[WQ Projection]
        RMSNorm1 --> WK[WK Projection]
        RMSNorm1 --> WV[WV Projection]
        
        WQ --> RoPE_Q[Apply Rotary Embeddings]
        WK --> RoPE_K[Apply Rotary Embeddings]
        
        RoPE_K --> K_Cache["Store/Retrieve KV Cache"]
        WV --> V_Cache["Store/Retrieve KV Cache"]
        
        K_Cache --> Repeat_KV["Repeat KV Heads if GQA"]
        V_Cache --> Repeat_KV
        
        RoPE_Q --> MatMul_Softmax["Scaled Dot-Product & Softmax"]
        Repeat_KV --> MatMul_Softmax
        MatMul_Softmax --> WO[WO Output Projection]
    end
    
    Input --> Add1["+"]
    WO --> Add1
    
    Add1 --> RMSNorm2[RMSNorm]
    
    subgraph ffn ["SwiGLU Feed-Forward Network"]
        RMSNorm2 --> W1[W1 Projection]
        RMSNorm2 --> W3[W3 Projection]
        
        W1 --> SiLU["SiLU / Swish Gate"]
        SiLU --> GateMul["*"]
        W3 --> GateMul
        GateMul --> W2[W2 Output Projection]
    end
    
    Add1 --> Add2["+"]
    W2 --> Add2
    
    Add2 --> Output["Output Tensor: out <br>Shape: B, Seq_Len, Dim"]
    
    style self_attn fill:#f3f6fa,stroke:#1a5fb4,stroke-width:1px
    style ffn fill:#f9f0ff,stroke:#613583,stroke-width:1px
    style Add1 fill:#d5fadc,stroke:#26a269,stroke-width:1px
    style Add2 fill:#d5fadc,stroke:#26a269,stroke-width:1px
```

---

## Core Component Visualizations

### 1. Rotary Position Embeddings (RoPE)
Instead of adding positional vectors to embeddings, RoPE applies a coordinate rotation to the Query ($Q$) and Key ($K$) representations in 2D slices. This allows the inner product of Query-Key pairs to decay with relative distance.

```mermaid
graph TD
    InputX["Input Tensor: Q / K <br>Shape: B, Seq_Len, H, Head_Dim"] --> SplitReal["Re-shape & View as Complex <br>x_complex Shape: B, Seq_Len, H, Head_Dim/2"]
    
    m["Token Position: m"] --> OuterProd["Outer Product: m * theta"]
    ThetaFreq["Theta Frequencies"] --> OuterProd
    OuterProd --> PolarForm["Polar Representation <br>freqs_complex Shape: Seq_Len, Head_Dim/2"]
    PolarForm --> Expand["Expand Dimensions <br>Shape: 1, Seq_Len, 1, Head_Dim/2"]
    
    SplitReal --> ComplexMul["*"]
    Expand --> ComplexMul
    
    ComplexMul --> ViewReal["View as Real Numbers <br>Shape: B, Seq_Len, H, Head_Dim/2, 2"]
    ViewReal --> Output["Reshape to Original Shape <br>Shape: B, Seq_Len, H, Head_Dim"]
    
    style PolarForm fill:#fff3cd,stroke:#ffc107,stroke-width:1px
    style ComplexMul fill:#cfe2ff,stroke:#0d6efd,stroke-width:1px
```

### 2. Grouped-Query Attention (GQA)
In vanilla multi-head attention (MHA), every query head has its own key-value head. Multi-query attention (MQA) shares a single key-value head across all query heads. Grouped-Query Attention (GQA) is an optimal middle-ground: it splits query heads into $G$ groups, and each group shares unique key-value heads.

This is executed using the `repeat_kv` operation:

```mermaid
graph LR
    subgraph query_heads ["Query Heads (H_Q = 8)"]
        q1["Q-Head 1"]
        q2["Q-Head 2"]
        q3["Q-Head 3"]
        q4["Q-Head 4"]
        q5["Q-Head 5"]
        q6["Q-Head 6"]
        q7["Q-Head 7"]
        q8["Q-Head 8"]
    end

    subgraph kv_heads ["KV Heads (H_KV = 2)"]
        kv1["KV-Head 1"]
        kv2["KV-Head 2"]
    end

    q1 -->|Repeat Group 1| kv1
    q2 -->|Repeat Group 1| kv1
    q3 -->|Repeat Group 1| kv1
    q4 -->|Repeat Group 1| kv1

    q5 -->|Repeat Group 2| kv2
    q6 -->|Repeat Group 2| kv2
    q7 -->|Repeat Group 2| kv2
    q8 -->|Repeat Group 2| kv2
    
    style kv1 fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
    style kv2 fill:#fff3e0,stroke:#ff9800,stroke-width:2px
```

---

## Repository Structure

*   [`model.py`](file:///home/esraa/LLaMa/model.py): Core LLaMA architectural modules (`Transformer`, `EncoderBlock`, `SelfAttention`, `FeedForward`, `RMSNorm`, RoPE helper functions).
*   [`inference.py`](file:///home/esraa/LLaMa/inference.py): Text generation pipeline helper wrapping the tokenizer, top-p sampling, and weights loader.
*   [`verify_llama.py`](file:///home/esraa/LLaMa/verify_llama.py): Unit test and mock evaluation pipeline for verifying model structures without heavy dependencies.

---

## Getting Started

### Prerequisites

*   Python 3.8+
*   PyTorch 2.0+
*   SentencePiece

```bash
pip install torch sentencepiece tqdm
```

### Running Validation Tests
You can verify the correctness of the custom implementation (RMSNorm, Attention cache mechanism, Rotary Positional Embeddings) by running:

```bash
python verify_llama.py
```

### Loading LLaMA 2 Weights and Generating Text
Make sure you have downloaded the weights from Meta (or Hugging Face) and converted them to standard formats:

```python
from inference import LLaMA

model = LLaMA.build(
    load_model=True,
    tokenizer_path="path/to/tokenizer.model",
    checkpoints_dir="path/to/llama-2-7b/",
    max_seq_len=1024,
    max_batch_size=4,
    device="cuda"
)

out_tokens, out_texts = model.text_completion(
    prompts=["The capital of France is"],
    max_gen_len=64
)
print(out_texts[0])
```
