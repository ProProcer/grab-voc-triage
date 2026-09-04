from google_play_scraper import Sort, reviews
import pandas as pd

app_id = 'com.grabtaxi.passenger'

result, _ = reviews(
    app_id,
    lang='id',              # 'id' for Indonesian, 'ms' for Malay
    country='id',           # 'id' or 'my'
    sort=Sort.NEWEST,
    count=10000,            # Pull 10k reviews
    filter_score_with=None  # Or pass 1, 2, 3 to grab low ratings directly
)

df = pd.DataFrame(result)[['userName', 'score', 'at', 'content']]
df.to_csv('data/raw/grab_reviews.csv', index=False)  