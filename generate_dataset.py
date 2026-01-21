import pickle
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from itertools import combinations
from time import time
from collections import OrderedDict
import nltk
import os
os.makedirs("Dataset", exist_ok=True)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Required for some versions of wordnet



# NLP Preprocessing Setup
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')

# Load scraped symptom dictionary
with open('final_dis_symp.pickle', 'rb') as handle:
    dis_symp = pickle.load(handle)

t0 = time()
total_symptoms = set()
diseases_symptoms_cleaned = OrderedDict()

# Clean and tokenize each disease's symptoms
for disease in sorted(dis_symp.keys()):
    raw_symptoms = re.sub(r"\[\S+\]", "", dis_symp[disease]).lower().split(',')
    clean_symptoms = []

    for sym in raw_symptoms:
        sym = sym.strip()
        if sym and sym != "none":
            sym = sym.replace('-', ' ').replace("'", '').replace('(', '').replace(')', '')
            tokens = splitter.tokenize(sym)
            filtered = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and not word[0].isdigit()]
            final_sym = ' '.join(filtered)
            total_symptoms.add(final_sym)
            clean_symptoms.append(final_sym)

    if clean_symptoms:
        diseases_symptoms_cleaned[disease] = clean_symptoms

print(f"Total diseases processed: {len(diseases_symptoms_cleaned)}")
print(f"Time for preprocessing symptoms: {round(time() - t0, 2)}s")

# Create one-hot encoding structure
symptom_list = ['label_dis'] + sorted(total_symptoms)
df_norm_rows = []
df_comb_rows = []

# Convert each disease's symptoms to binary feature rows
for disease, symptoms in diseases_symptoms_cleaned.items():
    disease = disease.encode().decode('utf-8')

    # Normal row: all symptoms present
    row_norm = dict.fromkeys(symptom_list, 0)
    row_norm['label_dis'] = disease
    for sym in symptoms:
        row_norm[sym] = 1
    df_norm_rows.append(row_norm)

    # Combination rows: all non-empty subsets of symptoms
    for r in range(1, len(symptoms) + 1):
        for subset in combinations(symptoms, r):
            row_comb = dict.fromkeys(symptom_list, 0)
            row_comb['label_dis'] = disease
            for sym in subset:
                row_comb[sym] = 1
            df_comb_rows.append(row_comb)

# Create final DataFrames
df_norm = pd.DataFrame(df_norm_rows)
df_comb = pd.DataFrame(df_comb_rows)

print(f"Shape of Normal Dataset: {df_norm.shape}")
print(f"Shape of Combination Dataset: {df_comb.shape}")
print(f"Time to convert to dataset: {round(time() - t0, 2)}s")

# Save to CSV
df_norm.to_csv("./Dataset/disease_symptom_matrix.csv", index=False)
df_comb.to_csv("./Dataset/disease_symptom_combinations.csv", index=False)

# Save human-readable symptom dictionary
with open('./Dataset/cleaned_disease_symptom_list.txt', 'w', encoding='utf-8') as f:
    for disease, symptoms in diseases_symptoms_cleaned.items():
        print([disease] + symptoms, file=f)