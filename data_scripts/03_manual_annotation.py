import config
import pandas as pd

df = pd.read_csv(config.CLEANED_DATA_PATH)
selected = df.groupby('score').sample(frac = 50.2 / len(df), random_state = config.RANDOM_STATE)
selected.to_csv(config.MANUAL_ANNOTATION_DATA_PATH)
