APP_ID = 'com.grabtaxi.passenger'

RAW_DATA_PATH = 'data/raw/grab_reviews.csv'
CLEANED_DATA_PATH = 'data/interim/grab_reviews.csv'
EXCLUDE_MANUAL_DATA_PATH = 'data/processed/grab_reviews.csv'

MANUAL_ANNOTATION_COUNT = 50.2
MANUAL_ANNOTATION_TEMPLATE_DATA_PATH = 'data/interim/manual_annotation.xlsx'
MANUAL_ANNOTATION_DATA_PATH = 'data/processed/manual_annotation.csv'
MANUAL_ANNOTATION_EXCEL_DATA_PATH = 'data/processed/manual_annotation.xlsx'

FEW_SHOT_COUNT = 10
FEW_SHOT_EXAMPLE_TEMPLATE_DATA_PATH = 'data/interim/few_shot_example.xlsx'
FEW_SHOT_EXAMPLE_DATA_PATH = 'data/processed/few_shot_example.md'
FEW_SHOT_EXAMPLE_EXCEL_DATA_PATH = 'data/processed/few_shot_example.xlsx'

OPENAI_BATCH_CLASSIFY_TEMP_JSONL = 'data/interim/batch_tasks.jsonl'
MODEL_PREDS_DIR = "data/preds"

VALID_SIZE = 0.2
PSEUDO_GT_DATA = "data/preds/openai/grab_reviews__ver2__gpt-4o-mini.csv"
TRAIN_PATH = 'data/processed/train.parquet'
VALID_PATH = 'data/processed/valid.parquet'
TEST_PATH = 'data/processed/test.parquet'

CATEGORIES = ['DRIVER_OPS', 'APP_AND_MAPS', 'PRICING_AND_BILLING']

RANDOM_STATE = 12