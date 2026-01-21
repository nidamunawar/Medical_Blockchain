import re
import random
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import spacy_stanza
from sklearn.metrics import precision_score, recall_score, f1_score, multilabel_confusion_matrix
from negspacy.negation import Negex
from negspacy.termsets import termset
from spacy.language import Language

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Load NLP Models
# -----------------------------
try:
    nlp = spacy.load("en_biobert_ner_symptom")
except OSError:
    raise RuntimeError("Model 'en_biobert_ner_symptom' not found. Please install it using: spacy download en_biobert_ner_symptom")

nlp_sz = spacy_stanza.load_pipeline("en", package="mimic", processors={"ner": "i2b2"}, use_gpu=False)

# Add Negex (Negation Detection)
ts = termset("en_clinical")
ts.add_patterns({
    'preceding_negations': ['abstain from', 'other than', 'except for', 'excluding', 'lacking', 'lack of', 'but', 'no', 'not'],
    'following_negations': ['negative', 'exclusionary']
})

@Language.factory("negex_legacy")
def create_negex_component(nlp, name):
    return Negex(
        nlp=nlp,
        name=name,
        neg_termset=ts.get_patterns(),
        ent_types=["PROBLEM", "TEST", "TREATMENT", "SYMPTOM"],
        extension_name="negex",
        chunk_prefix=["B"]
    )

nlp_sz.add_pipe("negex_legacy", last=True)

# -----------------------------
# Helper Functions
# -----------------------------

def chunk_text(text, max_tokens=384):
    """Split long text into smaller chunks to avoid transformer token limits."""
    words = text.split()
    chunks = []
    while words:
        chunk = []
        length = 0
        while words and (length + len(words[0])) <= max_tokens:
            word = words.pop(0)
            chunk.append(word)
            length += len(word)
        chunks.append(' '.join(chunk))
    return chunks


def apply(text):
    """Run BioBERT + NegEx pipeline on input text."""
    text = re.sub(r"[^A-Za-z.,:]", " ", text.strip())
    text = re.sub(r'[.]+', '.', text)
    text = re.sub(r'[,]+', ',', text)
    text = re.sub(r'\s+', ' ', text)

    chunks = chunk_text(text)
    entities = set()

    for chunk in chunks:
        doc = nlp(chunk)
        entities.update(ent.text.lower().strip() for ent in doc.ents)

        nd = nlp_sz(chunk)
        negated = {ent.text.lower().strip() for ent in nd.ents if hasattr(ent._, 'negex') and ent._.negex}
        entities.difference_update(negated)

    return list(entities) if entities else ["No symptom detected"]


# -----------------------------
# Input Data (Gold Standard)
# -----------------------------
input_sentences = [
    "I have chest pain and dizziness but no fever.",
    "Nausea and headache since morning.",
    "I am coughing but no pain.",
    "I feel fatigue and body ache.",
    "No signs of fever, but I’m experiencing sore throat.",
    "I'm having stomach cramps and nausea.",
    "Headache and blurred vision bothering me since night.",
    "Just a cough, nothing else.",
    "I have shortness of breath and chest pressure.",
    "Sneezing and runny nose for two days."
]

gold_symptoms = [
    ["chest pain", "dizziness"],
    ["nausea", "headache"],
    ["cough"],
    ["fatigue", "body ache"],
    ["sore throat"],
    ["stomach cramps", "nausea"],
    ["headache", "blurred vision"],
    ["cough"],
    ["shortness breath", "chest pressure"],
    ["sneezing", "runny nose"]
]


# -----------------------------
# Noise Injection
# -----------------------------

def introduce_noise(sentence):
    """Add typos, synonyms, and filler words to simulate noisy input."""
    typo_map = {
        "fever": "fevr",
        "headache": "headche",
        "nausea": "nausee",
        "fatigue": "fatique",
        "dizziness": "dizzi",
        "pain": "pian",
        "cough": "cof",
        "throat": "throt",
        "cramps": "crampz"
    }

    synonyms = {
        "dizziness": "lightheadedness",
        "shortness of breath": "breathing difficulty",
        "sore throat": "scratchy throat",
        "body ache": "muscle pain",
        "fatigue": "tiredness",
        "nausea": "queasiness",
        "runny nose": "nasal discharge",
        "cough": "dry cough"
    }

    fillers = ["kind of", "a bit of", "slightly", "pretty much", "you know", "some sort of"]

    words = sentence.split()
    new_words = []

    for w in words:
        # Random typo or synonym substitution
        if random.random() < 0.2:
            lw = w.lower()
            if lw in typo_map:
                w = typo_map[lw]
            elif lw in synonyms:
                w = synonyms[lw]
        new_words.append(w)

        # Random filler insertion
        if random.random() < 0.1:
            new_words.append(random.choice(fillers))

    # Add punctuation irregularities
    noisy_sentence = " ".join(new_words)
    noisy_sentence = re.sub(r'(\w)(\s)(\w)', lambda m: m.group(1) + random.choice([" ", "  ", " ... "]) + m.group(3), noisy_sentence)

    return noisy_sentence.strip()


# Optional: make reproducible
random.seed(42)
noisy_sentences = [introduce_noise(s) for s in input_sentences]

print("\n🧪 Sample Noisy Sentences:")
for i, ns in enumerate(noisy_sentences[:5]):
    print(f"{i+1}. {ns}")


# -----------------------------
# Evaluation
# -----------------------------
symptom_vocab = sorted(list({sym for row in gold_symptoms for sym in row}))

def encode_vector(symptoms, vocab):
    return [1 if sym in symptoms else 0 for sym in vocab]

y_true, y_pred, records = [], [], []

for i, sentence in enumerate(noisy_sentences):
    gold = gold_symptoms[i]
    pred = apply(sentence)
    y_true.append(encode_vector(gold, symptom_vocab))
    y_pred.append(encode_vector(pred, symptom_vocab))
    records.append({
        "Input Sentence": sentence,
        "Gold Symptoms": "; ".join(gold),
        "Predicted Symptoms": "; ".join(pred)
    })

# -----------------------------
# Metrics
# -----------------------------
y_true = np.array(y_true)
y_pred = np.array(y_pred)

precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
accuracy = (y_true == y_pred).mean()

print("\n✅ Evaluation Complete (Noisy Data)")
print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1 Score  : {f1:.2f}")

# -----------------------------
# Save CSV
# -----------------------------
df = pd.DataFrame(records)
df["Accuracy"] = accuracy
df["Precision"] = precision
df["Recall"] = recall
df["F1 Score"] = f1
df.to_csv("symptom_extraction_eval_with_noise.csv", index=False)
print("\n📁 Saved to: symptom_extraction_eval_with_noise.csv")

# -----------------------------
# Visualization
# -----------------------------
metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}
plt.figure(figsize=(6, 4))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="viridis")
plt.ylim(0, 1)
plt.title("Symptom Extraction Evaluation Metrics (Noisy Data)")
plt.ylabel("Score")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("metric_bar_chart_noisy.png")
plt.show()

# Confusion matrices per symptom
conf_matrices = multilabel_confusion_matrix(y_true, y_pred)

for idx, sym in enumerate(symptom_vocab):
    cm = conf_matrices[idx]
    plt.figure(figsize=(3, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    plt.title(f"Confusion Matrix - {sym}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{sym.replace(' ', '_')}_noisy.png")
    plt.show()
