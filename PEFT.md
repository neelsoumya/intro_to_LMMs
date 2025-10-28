# Parameter-Efficient Fine-Tuning (PEFT) for LLMs

## What is PEFT?

Parameter-Efficient Fine-Tuning (PEFT) is a set of techniques that allow you to adapt large language models to specific tasks while only training a small fraction of the model's parameters. Instead of fine-tuning all billions of parameters in an LLM, PEFT methods update only a tiny subset (often <1% of parameters), dramatically reducing:

- **Computational costs** (memory and GPU requirements)
- **Training time**
- **Storage requirements** (you only save the small adapter weights)

## Why PEFT Matters

Traditional full fine-tuning of a 7B parameter model requires:
- Significant GPU memory (often 80GB+ for training)
- Long training times
- Storing separate full model copies for each task

PEFT fine-tuning of the same model requires:
- Much less GPU memory (can fit on consumer GPUs)
- Faster training (hours instead of days)
- Only storing small adapter files (a few MB instead of GBs)

## Common PEFT Methods

### 1. **LoRA (Low-Rank Adaptation)**
The most popular PEFT method. LoRA freezes the pretrained model weights and injects trainable low-rank decomposition matrices into each layer.

**Key idea**: Instead of updating weight matrix W, add a low-rank update: `W' = W + BA`, where B and A are much smaller matrices.

**Advantages**:
- Very parameter efficient (0.1-1% of original parameters)
- No inference latency
- Multiple adapters can be swapped easily

### 2. **Prefix Tuning**
Adds trainable "prefix" vectors to the input of each transformer layer, keeping the original model frozen.

### 3. **Prompt Tuning**
Similar to prefix tuning but only adds trainable tokens to the input layer.

### 4. **Adapter Layers**
Inserts small trainable neural network modules (adapters) between frozen transformer layers.

## How LoRA Works (Detailed)

```
Original Forward Pass:
Input → Weight Matrix (W) → Output

LoRA Forward Pass:
Input → [Frozen W + Trainable (B×A)] → Output
         └─────────┬─────────┘
              Low-rank update
```

**Example dimensions**:
- Original weight matrix W: 4096 × 4096 = 16.7M parameters
- LoRA matrices: B (4096 × 8) + A (8 × 4096) = 65K parameters
- **Reduction**: 99.6% fewer parameters!

## Visual Architecture Comparison

```
┌─────────────────────────────────────────────────────────────┐
│                    FULL FINE-TUNING                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Embeddings          [ALL TRAINABLE ✓]               │
│        ↓                                                    │
│  Transformer Layer 1       [ALL TRAINABLE ✓]               │
│        ↓                                                    │
│  Transformer Layer 2       [ALL TRAINABLE ✓]               │
│        ↓                                                    │
│       ...                  [ALL TRAINABLE ✓]               │
│        ↓                                                    │
│  Transformer Layer N       [ALL TRAINABLE ✓]               │
│        ↓                                                    │
│  Output Layer              [ALL TRAINABLE ✓]               │
│                                                             │
│  Total Trainable: 100% of parameters (e.g., 7B params)     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    PEFT (LoRA Example)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Embeddings          [FROZEN ❄]                      │
│        ↓                                                    │
│  ┌──────────────────────────────────────┐                  │
│  │ Transformer Layer 1      [FROZEN ❄]  │                  │
│  │    ↓                                  │                  │
│  │    W_q [FROZEN] + LoRA_q [TRAIN ✓]  │                  │
│  │    W_k [FROZEN] + LoRA_k [TRAIN ✓]  │                  │
│  │    W_v [FROZEN] + LoRA_v [TRAIN ✓]  │                  │
│  └──────────────────────────────────────┘                  │
│        ↓                                                    │
│  ┌──────────────────────────────────────┐                  │
│  │ Transformer Layer 2      [FROZEN ❄]  │                  │
│  │    + LoRA adapters       [TRAIN ✓]   │                  │
│  └──────────────────────────────────────┘                  │
│        ↓                                                    │
│       ...                                                   │
│        ↓                                                    │
│  Transformer Layer N + LoRA [FROZEN ❄ + TRAIN ✓]          │
│        ↓                                                    │
│  Output Layer              [FROZEN ❄]                      │
│                                                             │
│  Total Trainable: 0.1-1% of parameters (e.g., 7-70M)       │
└─────────────────────────────────────────────────────────────┘
```

## LoRA Matrix Decomposition Detail

```
┌────────────────────────────────────────────────────────┐
│         How LoRA Modifies a Weight Matrix              │
└────────────────────────────────────────────────────────┘

Original Weight Matrix (W):
┌─────────────────────┐
│ d × d (e.g. 4096²)  │  ← FROZEN ❄
│                     │
│   [Large Matrix]    │
│                     │
└─────────────────────┘

LoRA Low-Rank Update:
        ┌──────┐      ┌──────────────┐
        │  B   │  ×   │      A       │
        │ d×r  │      │     r×d      │
        │      │      │              │
        └──────┘      └──────────────┘
           ↑                ↑
      r = 8-64        Rank (small!)
     [TRAINABLE ✓]   [TRAINABLE ✓]

Final Computation:
Output = Input × (W + α/r × BA)
                  ↑       ↑
              Frozen  Trainable
              
Where α is a scaling factor
```

## Practical Use Cases

1. **Task-specific adaptation**: Fine-tune for sentiment analysis, summarization, Q&A
2. **Domain adaptation**: Adapt general models to medical, legal, or technical domains
3. **Multi-task learning**: Train separate small adapters for different tasks, keep one base model
4. **Personalization**: Create user-specific adapters without full model copies

## Implementation Example (Conceptual)

```python
# Using Hugging Face PEFT library
from peft import LoraConfig, get_peft_model

# Configure LoRA
config = LoraConfig(
    r=8,                      # Rank
    lora_alpha=32,            # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.1,
)

# Apply to model
model = get_peft_model(base_model, config)

# Now only train ~0.1% of parameters!
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

## Key Takeaways

- PEFT enables efficient LLM adaptation with minimal resources
- LoRA is the most popular method due to its simplicity and effectiveness
- You can train on consumer hardware and store multiple adapters cheaply
- Performance is comparable to full fine-tuning for many tasks
- Essential for democratizing LLM customization
