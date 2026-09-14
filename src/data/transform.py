import pandas as pd
from datasketch import MinHashLSH
from src.data.utils import normalize_text, get_minhash
import config

def score_filter(df : pd.DataFrame) -> pd.DataFrame:
    df_filtered = df[df.score.between(1, 3)]
    df_top_star = df[df.score.between(4, 5)].sample(frac = 0.1, random_state= 32)
    return pd.concat(
        (df_filtered, df_top_star), ignore_index= True
    )

def length_filter(df : pd.DataFrame) -> pd.DataFrame:
    review_length = df.content.str.split().apply(lambda x : len(x))
    return df[review_length >= 6].reset_index(drop = True)


def normalize_review(df : pd.DataFrame) -> pd.DataFrame:
    df['content'] = df['content'].apply(lambda x : normalize_text(x))
    return df

def drop_duplicate_review(df : pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset = ['content']).reset_index(drop = True)

    minhashes = df['content'].apply(lambda x : get_minhash(x, n_gram = 3))

    lsh = MinHashLSH(threshold = 0.85, num_perm = 128)

    unique_indices = []

    for idx, m in enumerate(minhashes):
        matches = lsh.query(m)
        if not matches:
            lsh.insert(f'rev_{idx}', m)
            unique_indices.append(idx)
    
    return df.loc[unique_indices].reset_index(drop = True)

def encode_labels(df : pd.DataFrame) -> pd.DataFrame:
    df[config.CATEGORIES] = df[config.CATEGORIES].replace({'ABSENT' : 0, 'NEG' : 1}).astype('int64')
    return df