import tweepy
from textblob import TextBlob

# Authenticate with the Twitter API
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Search for tweets containing the keyword "python"
tweets = tweepy.Cursor(api.search, q='python').items(100)

# Perform sentiment analysis on each tweet
for tweet in tweets:
    text = tweet.text
    analysis = TextBlob(text)
    print(text, analysis.sentiment)