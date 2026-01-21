import os
import re
import joblib
import pandas as pd
import spacy
import spacy_stanza
from negspacy.termsets import termset
from negspacy.negation import Negex
from spacy.language import Language
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from itertools import combinations
import nltk
import multichain

# ------------------ Setup ------------------
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')

# Load model and dataset
model = joblib.load("Models/stacking_model_logistic_regression.joblib")
df_norm = pd.read_csv("Dataset/disease_symptom_matrix.csv")
dataset_symptoms = list(df_norm.columns[1:])

# Load BioBERT
try:
    nlp_biobert = spacy.load("en_biobert_ner_symptom")
except:
    raise RuntimeError("Missing model: en_biobert_ner_symptom")

# Load Stanza + negation detection
nlp_stanza = spacy_stanza.load_pipeline("en", package="mimic", processors={"ner": "i2b2"}, use_gpu=False)
ts = termset("en_clinical")
ts.add_patterns({
    'preceding_negations': ['abstain from', 'other than', 'except for', 'excluding', 'lacking', 'lack of', 'but', 'no', 'not'],
    'following_negations': ['negative', 'exclusionary']
})

@Language.factory("negex_legacy")
def create_negex_component(nlp, name):
    return Negex(nlp=nlp, name=name, neg_termset=ts.get_patterns(),
                 ent_types=["PROBLEM", "TEST", "TREATMENT", "SYMPTOM"],
                 extension_name="negex", chunk_prefix=["B"])

nlp_stanza.add_pipe("negex_legacy", last=True)

# ------------------ Functions ------------------

def chunk_text(text, max_tokens=384):
    words = text.split()
    chunks = []
    while words:
        current_chunk = []
        current_length = 0
        while words and (current_length + len(words[0])) <= max_tokens:
            word = words.pop(0)
            current_chunk.append(word)
            current_length += len(word)
        chunks.append(' '.join(current_chunk))
    return chunks

def extract_symptoms(text):
    text = re.sub(r"[^A-Za-z.,:]", " ", text.strip())
    text = re.sub(r'[.]+', '.', text)
    text = re.sub(r'[,]+', ',', text)
    text = re.sub(r'\s+', ' ', text)
    chunks = chunk_text(text)
    entities = set()
    for chunk in chunks:
        doc = nlp_biobert(chunk)
        entities.update({ent.text.lower().strip() for ent in doc.ents})
        nd = nlp_stanza(chunk)
        negated = {ent.text.lower().strip() for ent in nd.ents if hasattr(ent._, 'negex') and ent._.negex}
        entities.difference_update(negated)
    return list(entities) if entities else ["No symptom detected"]

def clean(sym):
    tokens = splitter.tokenize(sym.strip().replace('-', ' ').replace("'", ''))
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])

# ------------------ Patient Input ------------------

user_text = input("🩺 Please describe your symptoms:\n")
extracted = extract_symptoms(user_text)

if extracted == ["No symptom detected"]:
    print("\n❌ No symptoms detected.")
    exit()

processed_user_symptoms = [clean(sym) for sym in extracted]
matched_symptoms = set()
for data_sym in dataset_symptoms:
    for user_sym in processed_user_symptoms:
        match_count = sum(1 for token in data_sym.split() if token in user_sym.split())
        if match_count / len(data_sym.split()) > 0.5:
            matched_symptoms.add(data_sym)

if not matched_symptoms:
    print("\n⚠️ No matching symptoms found.")
    exit()

matched_symptoms = list(matched_symptoms)
print("\n🧠 Matched Symptoms:")
for i, sym in enumerate(matched_symptoms):
    print(f"{i}: {sym}")

selected_indices = input("\n✔️ Select symptoms by index (space-separated):\n").split()
final_symptoms = [matched_symptoms[int(idx)] for idx in selected_indices]

# Co-occurring suggestions
disease_set = set()
for sym in final_symptoms:
    disease_set.update(df_norm[df_norm[sym] == 1]['label_dis'])

counter = []
for dis in disease_set:
    row = df_norm[df_norm['label_dis'] == dis].iloc[0, 1:].values.tolist()
    for i, val in enumerate(row):
        if val == 1 and dataset_symptoms[i] not in final_symptoms:
            counter.append(dataset_symptoms[i])

symptom_counts = Counter(counter)
common_suggestions = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)

buffered = []
for i, (sym, _) in enumerate(common_suggestions, 1):
    buffered.append(sym)
    if i % 5 == 0 or i == len(common_suggestions):
        print("\n💡 Do you have any of these symptoms?")
        for j, s in enumerate(buffered):
            print(f"{j}: {s}")
        user_more = input("Enter indices (space-separated), 'no' to stop, '-1' to skip:\n").lower().split()
        if user_more[0] == 'no':
            break
        elif user_more[0] == '-1':
            buffered = []
            continue
        for idx in user_more:
            final_symptoms.append(buffered[int(idx)])
        buffered = []

# Vectorize
sample_x = [0] * len(dataset_symptoms)
print("\n✅ Final Symptoms Used for Prediction:")
for sym in final_symptoms:
    print(f"• {sym}")
    if sym in dataset_symptoms:
        sample_x[dataset_symptoms.index(sym)] = 1

# Predict disease
predicted_disease = model.predict([sample_x])[0]
print(f"\n🔮 Predicted Disease: {predicted_disease}")

# ------------------ CNIC + Blockchain Publishing ------------------

# Get CNIC and name
while True:
    cnic = input("\n🆔 Enter your CNIC (xxxxx-xxxxxxx-x): ").strip()
    if re.match(r"^\d{5}-\d{7}-\d{1}$", cnic):
        break
    print("❌ Invalid CNIC format. Please enter as xxxxx-xxxxxxx-x")

patient_name = input("🧾 Enter your Name: ").strip().lower().replace(" ", "_")

# Define streams
main_stream = "predicted_diseases"
cnic_stream = f"patient_{cnic}"

try:
    mc = multichain.MultiChainClient("127.0.0.1", 7354, "multichainrpc", "4qRiY8xGE5f3LhjTGmS4FUguCL77fU393kpzWS2jHMbN")
    print("🔗 Connected to MultiChain.")

    for stream in [main_stream, cnic_stream]:
        if not any(s["name"] == stream for s in mc.liststreams()):
            mc.create("stream", stream, False)
            print(f"✅ Created stream: {stream}")
        mc.subscribe(stream)

    record = {
        "cnic": cnic,
        "patient_name": patient_name,
        "user_text": user_text,
        "final_symptoms": final_symptoms,
        "predicted_disease": predicted_disease
    }

    mc.publish(main_stream, cnic, {"json": record})
    mc.publish(cnic_stream, "disease_record", {"json": record})
    print(f"📬 Published to:\n  • General stream: '{main_stream}'\n  • CNIC stream: '{cnic_stream}'")

except Exception as e:
    print(f"❌ Blockchain error: {e}")
