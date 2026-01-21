import spacy
import re
import spacy_stanza
from negspacy.termsets import termset
from negspacy.negation import Negex
from spacy.language import Language
import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load BioBERT-based symptom NER model
try:
    nlp = spacy.load("en_biobert_ner_symptom")
except OSError:
    raise RuntimeError("Model 'en_biobert_ner_symptom' not found. Please install it before running.")

# Load Stanza-based clinical NER pipeline with i2b2 model
nlp_sz = spacy_stanza.load_pipeline(
    "en",
    package="mimic",
    processors={"ner": "i2b2"},
    use_gpu=False
)

# Define custom termset for negation detection
ts = termset("en_clinical")
ts.add_patterns({
    'preceding_negations': ['abstain from', 'other than', 'except for', 'excluding', 'lacking', 'lack of', 'but', 'no', 'not'],
    'following_negations': ['negative', 'exclusionary']
})

# Create SpaCy pipeline component for legacy Negex
@Language.factory("negex_legacy")
def create_negex_component(nlp, name):
    return Negex(
        nlp=nlp,
        name=name,
        neg_termset=ts.get_patterns(),
        ent_types=["PROBLEM", "TEST", "TREATMENT", "SYMPTOM"],  # Include all possibly returned labels
        extension_name="negex",
        chunk_prefix=["B"]
    )

# Add negation detection to the Stanza pipeline
nlp_sz.add_pipe("negex_legacy", last=True)

# Helper function to split text into chunks
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

# Function to apply both NER and negation filtering
def apply(text):
    # Clean text but keep original casing (important for BioBERT)
    text = re.sub(r"[^A-Za-z.,:]", " ", text.strip())
    text = re.sub(r'[.]+', '.', text)
    text = re.sub(r'[,]+', ',', text)
    text = re.sub(r'\s+', ' ', text)

    chunks = chunk_text(text)
    entities = set()

    for chunk in chunks:
        # Run BioBERT model
        doc = nlp(chunk)
        print("BioBERT:", [(ent.text, ent.label_) for ent in doc.ents])
        entities_chunk = {ent.text for ent in doc.ents}
        entities.update(entities_chunk)

        # Run Stanza NER with negation detection
        nd = nlp_sz(chunk)
        print("Stanza:", [(ent.text, ent.label_, getattr(ent._, "negex", None)) for ent in nd.ents])
        negated_entities = {ent.text for ent in nd.ents if hasattr(ent._, 'negex') and ent._.negex}
        entities.difference_update(negated_entities)

    return list(entities) if entities else ["No symptom detected"]

# --------------------
# Test Block (Put at the end of your script)
# --------------------
if __name__ == "__main__":
    sample_text = "The patient reports dizziness, nausea and shortness of breath. No signs of fever or cough or vomiting."
    results = apply(sample_text)
    print("\nDetected Symptoms:", results)
