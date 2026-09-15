# HELP.md — Medical_Blockchain: Blockchain-Based Disease Prediction from Patient-Reported Symptoms

## Purpose of this document

This file orients anyone opening this repository for the first time — a
supervisor, a review committee, or a future contributor — without requiring a
separate walkthrough. It states what the system does, defines the technical
terms used elsewhere in the project, describes the dataset, and gives exact
steps to install, run, and reproduce results.

## Project overview

This project predicts a probable disease from a patient's free-text
description of their symptoms, and optionally records the prediction on a
blockchain for tamper-evident storage. It combines three components:

1. **Natural language symptom extraction** — a biomedical named-entity
   recognition (NER) model identifies symptom mentions in unstructured patient
   text, with negation detection to exclude symptoms the patient explicitly
   denies (e.g. "no fever").
2. **Disease prediction** — a stacked ensemble machine learning classifier,
   trained on a structured disease–symptom dataset, predicts the most likely
   disease from the confirmed symptom set.
3. **Blockchain record-keeping** — the prediction, together with patient
   identifiers, is optionally published to a MultiChain blockchain, giving
   each patient an append-only, auditable prediction history.

## Glossary of key terms

| Term | Definition |
|---|---|
| **NER (Named Entity Recognition)** | An NLP technique that locates and classifies spans of text into predefined categories — here, symptom mentions. |
| **BioBERT** | A BERT language model pretrained on biomedical text, used here (as `en_biobert_ner_symptom`) to recognise symptom entities. |
| **Negation detection (Negex)** | A rule-based method that determines whether a detected entity is stated as absent (e.g. "denies fever"), so negated symptoms are excluded rather than treated as present. |
| **One-hot encoding** | Representing each possible symptom as a binary column (1 = present, 0 = absent) so the dataset can be used by standard ML classifiers. |
| **Stacking / stacked ensemble** | An ensemble learning method that trains several base classifiers (here: Decision Tree, Random Forest, SVM) and combines their outputs through a meta-model (Logistic Regression, Random Forest, or Gradient Boosting) that makes the final prediction. |
| **Blockchain** | A distributed, append-only ledger; once a record is written, it cannot be silently altered — used here for auditability of predictions. |
| **MultiChain** | An open-source blockchain platform used in this project to store prediction records. |
| **Stream (MultiChain)** | A named, append-only data channel within a MultiChain blockchain; this project uses one shared stream for all predictions and one private stream per patient. |
| **CNIC** | Computerised National Identity Card number (Pakistan) — used here as the patient identifier for blockchain records. |
| **RPC (Remote Procedure Call)** | The protocol MultiChain exposes for external programs (like this project's Python client) to interact with the blockchain node. |

## System pipeline

```
Patient free-text symptoms
        │
        ▼
Symptom extraction (BioBERT NER + Stanza clinical NER + negation filtering)
        │
        ▼
Match against dataset symptom vocabulary → patient confirms/refines →
co-occurring symptoms suggested
        │
        ▼
Stacked ensemble classifier → predicted disease
        │
        ▼
(optional) Prediction record published to MultiChain blockchain
```

## Repository structure

```
Medical_Blockchain/
├── HELP.md                              Project documentation (this file)
├── generate_dataset.py                  Builds Dataset/ from final_dis_symp.pickle
├── symptom_extraction.py                NLP symptom extraction module
├── train_stacking_model.py              Trains and saves the stacking classifiers
├── optimize_GB.py                       Hyperparameter search for Gradient Boosting
├── conf_matrix_meta_models.py           Generates the confusion matrices below
├── eval_symptom_extraction.py           Precision / recall / F1 for the NLP stage
├── predict_disease_cli.py               Interactive CLI: symptoms → predicted disease
├── predict_disease_cli_multichain.py    Same, plus publishes the result to MultiChain
├── multichain.py                        MultiChain JSON-RPC client library
├── final_dis_symp.pickle                Raw disease → symptom-description data (input)
├── main.py                              PyCharm project boilerplate — not part of the
│                                          pipeline; safe to remove
├── Dataset/
│   ├── cleaned_disease_symptom_list.txt Human-readable cleaned symptom lists
│   ├── disease_symptom_matrix.csv       One row per disease, its full symptom set
│   └── disease_symptom_combinations.csv One row per symptom subset per disease
│                                          (the model training set)
└── Confusion_Matrices/
    ├── confusion_matrix_logistic_regression.png
    └── confusion_matrix_random_forest.png
```

Trained model files (`.joblib`, produced by `train_stacking_model.py`) are not
committed here — they are 429 MB and 724 MB, above GitHub's 100 MB per-file
limit. Regenerate them locally (see Usage).

## Dataset description

`final_dis_symp.pickle` contains disease-to-symptom-description pairs, the
raw input from which the structured dataset is derived. `generate_dataset.py`
cleans this text (tokenisation, lemmatisation, stopword removal) and produces
two one-hot encoded CSV files:

| File | Rows | Size | Purpose |
|---|---|---|---|
| `Dataset/disease_symptom_matrix.csv` | 1 per disease | 283 KB | Reference lookup of each disease's full symptom set |
| `Dataset/disease_symptom_combinations.csv` | 1 per non-empty symptom subset, per disease | ~9.9 MB | Model training and evaluation set |

In both files, column `label_dis` holds the disease name; every other column
is a cleaned symptom phrase, with a value of 1 where that symptom applies to
the row and 0 otherwise.

## Installation

This project needs two sets of dependencies: lightweight ones for dataset
generation, and heavier ones (spaCy models, Stanza, transformers) for NLP,
training, and prediction. Python 3.9 is recommended.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install numpy pandas scikit-learn xgboost nltk beautifulsoup4 requests joblib html5lib googlesearch-python
pip install spacy spacy-stanza negspacy transformers torch fastapi uvicorn
python -m spacy download en_biobert_ner_symptom
python -c "import stanza; stanza.download('en', package='mimic')"
```

(Consider committing a `requirements.txt` with pinned versions so results are
reproducible by others — this repository does not currently include one.)

## Usage

```bash
# 1. Build the structured dataset from the raw data
python generate_dataset.py

# 2. Train the disease-prediction models
python train_stacking_model.py

# 3. Optional: hyperparameter tuning and evaluation
python optimize_GB.py
python conf_matrix_meta_models.py
python eval_symptom_extraction.py

# 4. Run symptom-based disease prediction
python predict_disease_cli.py

# 5. Run prediction with blockchain recording enabled
python predict_disease_cli_multichain.py
```

Step 5 requires a running MultiChain node (`multichaind`), reachable at the
RPC host, port, and credentials configured in `predict_disease_cli_multichain.py`.
Installation instructions: https://www.multichain.com/download-community/

## Limitations and planned work

- The dependency list above includes `fastapi` and `uvicorn`, indicating a web
  API layer around the prediction pipeline is planned; it has not yet been
  implemented in this codebase.
- Trained model weights are excluded from version control due to file size;
  they must be regenerated locally.
- The web-scraping step that originally produced `final_dis_symp.pickle` is
  not included in this repository; only its output is.

## Security consideration

`predict_disease_cli_multichain.py` currently defines its MultiChain RPC
username and password as literal values in the source file. Since this
repository is now public, these credentials are publicly visible. They
should be rotated and moved to environment variables, for example:

```python
import os
mc = multichain.MultiChainClient(
    os.environ["MC_HOST"], int(os.environ["MC_PORT"]),
    os.environ["MC_USER"], os.environ["MC_PASS"]
)
```
