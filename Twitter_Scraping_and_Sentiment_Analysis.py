import pandas as pd
import itertools
import snscrape.modules.twitter as sntwitter
import plotly.graph_objects as go
from datetime import datetime

start_time = datetime.now()
limit = 1000
#Creating dataframe called 'data' and storing the tweets from May 1st 2021 to 30th Juy 2021 for 'Vaccine'
data = pd.DataFrame(itertools.islice(sntwitter.TwitterSearchScraper('"#vaccine since:2021-05-01 until:2021-07-30"').get_items(),limit))

end_time = datetime.now()

# Printing the time taken to scrape these tweets
print('Duuration: {} '.format(end_time - start_time))


df = data[['date', 'id', 'user', 'content']]
df1 = df.head(5000)

from transformers import pipeline
sentiment_classifier = pipeline('sentiment-analysis')

df1 = (
    df1
    .assign(sentiment = lambda x: x['content'].apply(lambda s: sentiment_classifier(s)))
    .assign(
         label = lambda x: x['sentiment'].apply(lambda s: (s[0]['label'])),
         score = lambda x: x['sentiment'].apply(lambda s: (s[0]['score']))
    )
)

df1.head(1000).to_csv("tweets.csv", index=False)




















