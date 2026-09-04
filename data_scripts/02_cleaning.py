import pandas as pd
from src.preprocessing.transform import score_filter, length_filter, normalize_review, drop_duplicate_review
import config

df = pd.read_csv(config.RAW_DATA_PATH)
df = (
    df
    .pipe(score_filter)
    .pipe(length_filter)
    .pipe(normalize_review)
    .pipe(drop_duplicate_review)
)
df.to_csv(config.CLEANED_DATA_PATH, index = False)