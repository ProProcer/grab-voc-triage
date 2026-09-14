import re
from datasketch import MinHash
import pandas as pd
import json

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

def df_to_few_shot_markdown(
    df: pd.DataFrame,
    review_col: str,
    label_cols: list[str],
    output_path: str
) -> str:
    """
    Converts a labeled DataFrame into a formatted Markdown Few-Shot section
    compatible with system_prompt.md.
    """
    if label_cols is None:
        label_cols = [
            "DRIVER_OPERATIONS",
            "APP_AND_MAPS",
            "PRICING_AND_BILLING",
            "FULFILLMENT_FOOD"
        ]

    markdown_lines = ["## 4. Few-Shot Demonstrations\n"]

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        # Sanitize review text: strip whitespace and escape internal double quotes
        cleaned_review = str(row[review_col]).strip().replace('"', '\\"')

        # Construct dictionary adhering strictly to the label order
        label_dict = {head: str(row[head]).strip().upper() for head in label_cols}
        formatted_json = json.dumps(label_dict, indent=2, ensure_ascii=False)

        # Build markdown block
        block = (
            f"### Example {i}\n"
            f'**Review:**\n"{cleaned_review}"\n\n'
            f"**Output:**\n"
            f"```json\n{formatted_json}\n```\n"
        )
        markdown_lines.append(block)

    full_markdown = "\n".join(markdown_lines)

    # Save to disk
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_markdown)

    return full_markdown