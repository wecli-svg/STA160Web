# run in colab
from google.colab import files
files.download('combined_CLEAN.csv')
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

df = pd.read_csv('combined_CLEAN.csv')

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    noise_phrases = ["consent of instructor", "upper division standing", "pass/no pass"]
    for phrase in noise_phrases:
        text = text.replace(phrase, "")
    return text


df["processed_text"] = (
    "Course: " + df["Title"].fillna("") + ". " +
    df["Title"].fillna("") + ". " +
    "Description: " + df["Course Description"].apply(clean_text)
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Using device: {device}")

model = SentenceTransformer('all-mpnet-base-v2', device=device)

embeddings = model.encode(
    df["processed_text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_tensor=True,
    normalize_embeddings=True
)


torch.save(embeddings.cpu(), 'course_embeddings.pt')

from google.colab import files
files.download('course_embeddings.pt')
