# Suicide Intention Detection — RoBERTa-CNN Model

A hybrid deep learning model combining **RoBERTa** (Transformer) and **CNN** for binary text classification (detecting suicide-related risk language vs. non-risk language).

---

## 🎯 Quick Start Demo

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Quick Demo (Recommended for First-Time Users)

```bash
# Run with sample dataset and --dry-run for quick test
python roberta_cnn_training_final.py --dry-run --no-wandb --epochs 1

# Or run full training on sample data
python roberta_cnn_training_final.py --no-wandb --epochs 5 --batch-size 8
```

### 3. Use Your Own Data

```bash
# Place your CSV file in this directory (must have 'text' and 'class'/'label' columns)
python roberta_cnn_training_final.py --data-path your_dataset.csv --epochs 10 --no-wandb
```

---

## 📊 Expected Output

```
Using device: cpu
Loading data from Suicide_Dataset_sample.csv...
Loaded 46 samples
Normalizing text...
Applying data augmentation...
Total samples after augmentation: 50
Loading RoBERTa tokenizer and model...
Splitting data...
Train: 36, Val: 5, Test: 5
Creating datasets with max_len=256...
Starting training for 1 epochs...
Epoch 1/1, Train Loss: 0.6932, Train Accuracy: 0.5556

Evaluating on test set...
Test Report:
              precision    recall  f1-score   support

           0       0.00      0.00      0.00         2
           1       0.60      1.00      0.75         3

    accuracy                           0.60         5
   macro avg       0.30      0.50      0.38         5
weighted avg       0.36      0.60      0.45         5
```

---

## 🛠️ Command-Line Arguments

```
--data-path TEXT           Path to dataset CSV (default: Suicide_Dataset_sample.csv)
--batch-size INT           Batch size for training (default: 32)
--epochs INT               Number of training epochs (default: 5)
--lr FLOAT                 Learning rate (default: 0.005)
--dry-run                  Run single epoch for testing
--no-wandb                 Disable Weights & Biases logging
--no-cuda                  Disable GPU (force CPU)
--save-model               Save trained model weights
--seed INT                 Random seed (default: 1)
-h, --help                 Show all arguments
```

### Example: Full Training with Model Saving

```bash
python roberta_cnn_training_final.py \
  --data-path my_data.csv \
  --epochs 20 \
  --batch-size 16 \
  --lr 0.0001 \
  --save-model \
  --no-wandb
```

---

## 📁 File Structure

```
toGithub/
├── roberta_cnn_training_final.py    # Main training script
├── Suicide_Dataset_sample.csv       # Sample data for demo
├── results_maxlen256.py             # Training results reference
├── Data_api.ipynb                   # Data API notebook (optional)
├── requirements.txt                 # Python dependencies
├── README_DEMO.md                   # This file
└── README.md                        # Project overview & ethical notes
```

---

## 📋 Data Format

Your CSV file should have the following columns:

| Column | Type | Values |
|--------|------|--------|
| `text` | string | The text to classify |
| `class` or `label` | string | `suicide` or `non-suicide` (or 0/1) |

**Example:**

```csv
text,class
i just want to end it all,suicide
i feel hopeless,suicide
i love this new project,non-suicide
the weather is beautiful,non-suicide
```

---

## 🏋️ Model Architecture

```
Input Text
    ↓
RoBERTa Tokenizer (max_len=256)
    ↓
RoBERTa-Base (768-dim embeddings)
    ↓
Conv1d (100 filters, kernel_size=2)
    ↓
ReLU + MaxPool
    ↓
Fully Connected (FC) Layer → Binary Classification
    ↓
Softmax → Sigmoid → Prediction
```

**Key Hyperparameters:**
- Max sequence length: 256
- RoBERTa variant: `roberta-base` (12 layers, 768 hidden size)
- CNN filters: 100
- CNN kernel size: 2
- Optimizer: Adam (lr=0.0001)
- Loss: Cross Entropy

---

## 📊 Monitoring Training

### With Weights & Biases (W&B)

```bash
# Install wandb
pip install wandb

# Login with your W&B account
wandb login

# Train with logging enabled (default when wandb is installed)
python roberta_cnn_training_final.py --epochs 20 --data-path your_data.csv
```

### Without W&B (Local Only)

```bash
python roberta_cnn_training_final.py --no-wandb --epochs 20
```

Metrics are printed to console in both cases.

---

## ⚠️ Important Ethical Considerations

This tool is designed for **research and educational purposes**. When deploying in real scenarios:

1. **Never rely solely on automation** — Always include human review by trained mental health professionals
2. **Transparency** — Clearly communicate to users that predictions are assistive, not diagnostic
3. **Privacy** — De-identify data and follow local data protection regulations (GDPR, HIPAA, etc.)
4. **Safeguards** — Implement rate limiting, logging, and clear escalation paths to crisis services
5. **Bias awareness** — The model reflects biases in training data; test across diverse populations
6. **Ongoing monitoring** — Track model performance over time and retrain as needed

---

## 🔍 Troubleshooting

### Error: "Suicide_Dataset_sample.csv not found"

**Solution:** Make sure you're running the script from the `toGithub/` directory.

```bash
cd toGithub/
python roberta_cnn_training_final.py --no-wandb --dry-run
```

### Error: Out of Memory (OOM)

**Solution:** Reduce batch size or max sequence length:

```bash
python roberta_cnn_training_final.py --batch-size 8 --no-wandb
```

### Error: "No module named 'transformers'"

**Solution:** Install dependencies:

```bash
pip install -r requirements.txt
```

### Slow training on CPU

**Solution:** This is expected. For faster training, use GPU if available. Check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📝 Results from Original Research

The model was trained on a larger dataset (Suicide_Dataset.csv) with 50 epochs:

```
Max-length 256:
- Train Accuracy: 99.79%
- Validation Accuracy: 97.51%
- Validation F1: 0.9464
- Validation Precision: 0.9232
- Validation Recall: 0.9707
```

Results on sample data will be different due to smaller training set.

---

## 📚 References

- Devlin et al. (2018) BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- Liu et al. (2019) RoBERTa: A Robustly Optimized BERT Pretraining Approach
- Hugging Face Transformers: https://huggingface.co/docs/transformers/

---

## 📧 Contact & Citation

For questions or to report issues, please refer to the main project documentation.

If you use this code in your research, please cite appropriately and acknowledge the original authors.

---

**Last Updated:** March 2026 | **Status:** Demo/Research Version
