import pandas as pd
import re
import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def clean_data_lemma(data):
    """
    All data to lower
    Keep only alphanumeric characters 
    """
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = data[col].astype(str).str.lower()
        data[col] = data[col].apply(remove_punctuation)
        data[col] = data[col].apply(lemmatize_text)

    return data
def clean_data(data):
    """
    All data to lower
    Keep only alphanumeric characters 
    """
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = data[col].astype(str).str.lower()
        data[col] = data[col].apply(remove_punctuation)
    return data

def remove_punctuation(s: str):
    s = re.sub(r"[^a-zA-Z\s-]", " ", s)   # replace with space
    s = re.sub(r"\s+", " ", s)            # replace all multiple space with single space
    return s.strip()

def lemmatize_text(s: str):
    doc = nlp(s)
    return " ".join([token.lemma_ for token in doc])

def split_tax(data):
    """
    splits the taxonomy into 'field' and 'what'
    stop_words_tax was deducted in the notebook after counting the most used words in taxonomy
    """
    stop_words_tax = ["services", "manufacturing", "installation", "and", "production", "residential", "commercial", "management", "processing", "consulting"]
    field = data.split()
    what = ""
    for word in stop_words_tax:
        if word in field:
            field.remove(word)
            what = word
    field = " ".join(field)
    return field, what