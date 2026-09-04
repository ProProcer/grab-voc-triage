import pandas as pd
from src.preprocessing.transform import score_filter, length_filter, normalize_review, drop_duplicate_review

df = pd.read_csv("data/raw/grab_reviews.csv")
df = (
    df
    .pipe(score_filter)
    .pipe(length_filter)
    .pipe(normalize_review)
    .pipe(drop_duplicate_review)
)
df.to_csv('data/processed/grab_reviews.csv', index = False)