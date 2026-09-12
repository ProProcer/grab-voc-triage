import pandas as pd
import argparse
import config
from src.data.transform import encode_labels
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type = str, default=config.PSEUDO_GT_DATA)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df[config.CATEGORIES] = df[[c + '_pred' for c in config.CATEGORIES]]
    df = df[["content"] + config.CATEGORIES]
    df = encode_labels(df)

    splitter = MultilabelStratifiedShuffleSplit(n_splits = 1, test_size = config.VALID_SIZE, random_state= config.RANDOM_STATE)
    for train_idx, valid_idx in splitter.split(df[['content']], df[config.CATEGORIES]):
        pass

    df_train = df.iloc[train_idx]
    df_valid = df.iloc[valid_idx]

    df_train.to_parquet(config.TRAIN_PATH, index = False)
    df_valid.to_parquet(config.VALID_PATH, index = False)
    
    df_test = pd.read_csv(config.MANUAL_ANNOTATION_DATA_PATH)
    df_test = df_test[["content"] + config.CATEGORIES]
    df_test = encode_labels(df_test)

    df_test.to_parquet(config.TEST_PATH, index = False)

if __name__ == "__main__":
    main()