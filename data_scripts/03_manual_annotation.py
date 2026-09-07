import config
import pandas as pd
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--force_template', action = 'store_true')
    parser.add_argument('-e', '--force_excel2csv', action = 'store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    if not Path(config.MANUAL_ANNOTATION_TEMPLATE_DATA_PATH).exists() or args.force_template:
        df = pd.read_csv(config.CLEANED_DATA_PATH)
        selected = df.groupby('score').sample(frac = 50.2 / len(df), random_state = config.RANDOM_STATE)
        selected.to_excel(config.MANUAL_ANNOTATION_TEMPLATE_DATA_PATH)
    if not Path(config.MANUAL_ANNOTATION_DATA_PATH).exists() or args.force_excel2csv:
        df_annotated = pd.read_excel(config.MANUAL_ANNOTATION_EXCEL_DATA_PATH, index_col = 0)
        df_annotated.index.name = None
        df_annotated.to_csv(config.MANUAL_ANNOTATION_DATA_PATH)

if __name__ == '__main__':
    main()