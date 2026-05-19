# Named Entity Recognition Models Documentation

This document describes the two NER models implemented in this project.

---

## Model 1: DistilBERT Token Classifier (Day5_Train.ipynb)

### Overview
A fine-tuned **DistilBERT-base-uncased** model for Named Entity Recognition, using the Hugging Face Transformers library.

### Architecture
- **Base Model**: DistilBERT-base-uncased (66M parameters)
- **Task**: Token Classification (NER)
- **Labels**: 32 entity types (BIO format)

### Key Components
| Component | Details |
|-----------|---------|
| Encoder | DistilBERT transformer encoder |
| Pooling | [CLS] token representation |
| Classifier | Linear layer over hidden states |
| Output | Per-token entity predictions |

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-5 |
| Epochs | 15 |
| Batch Size | 16 |
| Max Sequence Length | 128 |
| Warmup Steps | 50 |
| Weight Decay | 0.01 |
| Early Stopping | Patience = 5 |

### Data Split
- Training: 2,490 samples
- Validation: 311 samples
- Test: 312 samples

### Performance (Test Set)
| Metric | Score |
|--------|-------|
| **F1** | **0.8490** |
| Precision | 0.8590 |
| Recall | 0.8392 |

### Entity-Level Performance (Top)
| Entity | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| CARDINAL | 0.9032 | 0.9180 | 0.9106 |
| DATE | 0.9441 | 0.9783 | 0.9609 |
| MONEY | 0.9444 | 1.0000 | 0.9714 |
| GPE | 0.8921 | 0.9688 | 0.9288 |
| ORDINAL | 1.0000 | 1.0000 | 1.0000 |

### Prediction Examples
```
Input: "Joe Biden met Donald Trump in Washington D.C. on Monday"
Output: Joe -> B-PERSON, Biden -> I-PERSON, Donald -> B-PERSON, Trump -> I-PERSON,
        Washington -> B-GPE, D.C. -> I-GPE, Monday -> B-DATE

Input: "Apple Inc. announced $1 billion investment"
Output: Apple -> B-ORG, Inc. -> I-ORG, $1 -> B-MONEY, billion -> I-MONEY
```

### Files Generated
- Model: `D:/Python/NLP/CK/ner_model_final/`
- Label Mappings: `D:/Python/NLP/CK/label_mappings.json`

---

## Model 2: Improved BiLSTM with Attention (Day5_5_BiLSTM_CRF_Improved.ipynb)

### Overview
A custom **BiLSTM model** with multi-head attention and layer normalization, built with TensorFlow/Keras.

### Architecture
```
Input (word_ids)
    ↓
Embedding (300d)
    ↓
Dropout (0.3)
    ↓
BiLSTM #1 (256 units) → LayerNorm → Dropout
    ↓
BiLSTM #2 (256 units) → LayerNorm → Dropout
    ↓
Multi-Head Attention (4 heads, key_dim=256)
    ↓
Dropout → Add & LayerNorm
    ↓
Dense (256) → Dropout
    ↓
Output Dense (num_tags)
```

### Key Components
| Component | Details |
|-----------|---------|
| Word Embedding | 300-dimensional, with masking |
| BiLSTM | 2 layers, 256 units each (Bidirectional) |
| Layer Normalization | After each LSTM block |
| Multi-Head Attention | 4 heads, 256 key dimensions |
| Dropout | 0.3 throughout |
| Output | Dense layer with softmax (logits) |

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Initial LR | 1e-3 |
| Epochs | 50 |
| Batch Size | 32 |
| Max Sequence Length | 128 |
| Early Stopping | Patience = 7 |
| Reduce LR | Patience = 3, factor = 0.5 |

### Data Split
- Training: ~80% (2,490 samples)
- Validation: 10% (311 samples)
- Test: 10% (312 samples)

### Saved Files
- Model: `D:/Python/NLP/CK/bilstm_crf_improved.keras`
- Label Mappings: `D:/Python/NLP/CK/label_mappings.json`

---

## Comparison

| Aspect | DistilBERT | BiLSTM + Attention |
|--------|------------|---------------------|
| **Parameters** | ~66M | ~5M (estimated) |
| **Type** | Pre-trained transformer | Custom neural network |
| **F1 Score** | 0.8490 | N/A (not evaluated in this notebook) |
| **Training Time** | Longer | Faster |
| **Hardware** | GPU recommended | Can run on CPU |
| **Inference Speed** | Slower | Faster |

---

## Entity Types (32 labels)

The models recognize the following NER categories:

- CARDINAL, DATE, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERSON, PRODUCT, QUANTITY, TIME, WORK_OF_ART

Each entity can have B- (Beginning), I- (Inside), or O (Outside) tags.

---

## Usage

### Using DistilBERT Model
```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

model = AutoModelForTokenClassification.from_pretrained('D:/Python/NLP/CK/ner_model_final')
tokenizer = AutoTokenizer.from_pretrained('D:/Python/NLP/CK/ner_model_final')

def predict(text):
    words = text.split()
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        preds = torch.argmax(model(**inputs).logits, dim=2)[0]
    # ... process predictions
    return predictions
```

### Using BiLSTM Model
```python
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('D:/Python/NLP/CK/bilstm_crf_improved.keras', compile=False)

def predict(text):
    words = text.split()
    token_ids = [word2id.get(w, word2id['<UNK>']) for w in words]
    token_ids += [word2id['<PAD>']] * (MAX_LEN - len(token_ids))
    logits = model.predict(np.array([token_ids]), verbose=0)[0]
    pred_ids = np.argmax(logits, axis=-1)
    return [(w, id2label.get(int(p), 'O')) for w, p in zip(words, pred_ids[:len(words)])]
```

---

## Recommendations

1. **Use DistilBERT** when:
   - Maximum accuracy is required
   - Sufficient GPU memory is available
   - Training time is not critical

2. **Use BiLSTM** when:
   - Lightweight deployment is needed
   - Faster inference is priority
   - Limited computational resources available

---

*Documentation generated: 2026-05-16*