import pandas as pd
import numpy as np

def load_data():
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="thyroid0387_UCI")
    return df

def preprocess_data(df):
    df_clean = df.drop(columns=['Record ID', 'referral source'], errors='ignore')
    df_encoded = pd.get_dummies(df_clean)
    df_encoded = df_encoded.fillna(0)
    return df_encoded

def get_document_vectors(df_encoded):
    doc1 = df_encoded.iloc[0].values
    doc2 = df_encoded.iloc[1].values
    return doc1, doc2

def cosine_similarity(doc1, doc2):
    dot_product = np.dot(doc1, doc2)
    norm_doc1 = np.linalg.norm(doc1)
    norm_doc2 = np.linalg.norm(doc2)
    if norm_doc1 == 0 or norm_doc2 == 0:
        return 0
    similarity = dot_product / (norm_doc1 * norm_doc2)
    return similarity


df = load_data()
df_encoded = preprocess_data(df)
doc1, doc2 = get_document_vectors(df_encoded)
cos_sim = cosine_similarity(doc1, doc2)


print("Cosine Similarity between the two document vectors:", cos_sim)
