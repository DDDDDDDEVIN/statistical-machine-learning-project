# Machine-Generated Text Detection with Domain-Aware Modeling

This project investigates the task of distinguishing human-written text from machine-generated content under realistic domain shift and class imbalance conditions. We explore both domain-specific and domain-unified modeling frameworks using a BERT-based encoder and evaluate the effectiveness of MMD-based alignment and semi-supervised learning.

## File Structure

- `bert_baseline.ipynb`: BERT model trained directly on TF-IDF features (baseline).
- `mlp_baseline.ipynb`: Simple MLP trained directly on TF-IDF features (baseline).
- `specific_mlp.ipynb`: Domain-specific pipeline using domain classifier and MLP classifiers.
- `specific_lstm.ipynb`: Same as above but with LSTM classifiers.
- `unified_mixup_semi.ipynb`: Unified model with semi-supervised learning and Mixup-based resampling. 
- `MAIN_unified_SMOTE_semi.ipynb`: Unified model with semi-supervised learning and SMOTE resampling. This is our final model design.
- `requirements.txt`: Python environment dependencies.
- `meeting minutes/`: Project logs or planning notes.

## Data

All input data files (i.e. domain1_train_data.json, domain2_train_data.json, test_data.json) should be placed in the root directory alongside the notebooks. No separate `data/` folder is required.

## Setup

Make sure you have Python ≥3.8 and install dependencies via:

```bash
pip install -r requirements.txt
```

We recommend using a GPU environment such as Google Colab or a local machine with CUDA support.

## Running Experiments

Simply open and run each notebook in order to reproduce results:

- Baselines:  
  `mlp_baseline.ipynb`, `bert_baseline.ipynb`

- Domain-Specific:  
  `specific_mlp.ipynb`, `specific_lstm.ipynb`

- Domain-Unified (with semi-supervised):  
  `MAIN_unified_mixup_semi.ipynb`, `unified_SMOTE_semi.ipynb`

## 📝 Notes

- All models are trained from scratch without pretraining.
- Evaluation is based on accuracy and F1-score on a balanced test set.