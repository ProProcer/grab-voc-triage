import config
import pandas as pd
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--force_template', action = 'store_true')
    parser.add_argument('-e', '--force_conversion', action = 'store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    if not (Path(config.MANUAL_ANNOTATION_TEMPLATE_DATA_PATH).exists() and Path(config.FEW_SHOT_EXAMPLE_TEMPLATE_DATA_PATH)) or args.force_template:
        df = pd.read_csv(config.CLEANED_DATA_PATH)
        df[config.CATEGORIES] = None
        selected = df.groupby('score').sample(frac = config.MANUAL_ANNOTATION_COUNT / len(df), random_state = config.RANDOM_STATE)
        selected.to_excel(config.MANUAL_ANNOTATION_TEMPLATE_DATA_PATH)

        df = df.drop(selected.index)
        selected_few_shot = df.groupby('score').sample(frac = config.FEW_SHOT_COUNT / len(df), random_state = config.RANDOM_STATE)
        selected_few_shot.to_excel(config.FEW_SHOT_EXAMPLE_TEMPLATE_DATA_PATH)

    if not Path(config.MANUAL_ANNOTATION_DATA_PATH).exists() or args.force_conversion:
        df_annotated = pd.read_excel(config.MANUAL_ANNOTATION_EXCEL_DATA_PATH, index_col = 0)
        df_annotated.index.name = None
        df_annotated.to_csv(config.MANUAL_ANNOTATION_DATA_PATH)

    if not Path(config.FEW_SHOT_EXAMPLE_DATA_PATH).exists() or args.force_conversion:
        df_annotated_few_shot = pd.read_excel(config.FEW_SHOT_EXAMPLE_EXCEL_DATA_PATH, index_col = 0)
        df_annotated_few_shot.index.name = None
        df_annotated_few_shot[['content'] + config.CATEGORIES].to_json(
            config.FEW_SHOT_EXAMPLE_DATA_PATH,
            orient='records',
            indent=2,
            force_ascii=False
        )
if __name__ == '__main__':
    main()