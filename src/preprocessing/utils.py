import re
from datasketch import MinHash

def normalize_text(text : str) -> str:
    text = str(text).lower()

    # remove URL and email pattern
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # collapse character elongation: 'paraaaaah' -> 'parah
    text = re.sub(r'(.)\1{2,}', '\1', text)

    # keep only alphanumeric and basic spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def get_minhash(text : str, n_gram : int = 3) -> MinHash:
    m = MinHash(num_perm = 128)

    shingles = [text[i : i + n_gram] for i in range(len(text) - n_gram + 1)]

    for s in set(shingles):
        m.update(s.encode('utf8'))

    return m