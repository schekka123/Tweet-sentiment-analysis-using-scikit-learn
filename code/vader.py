import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils import get_tweets

num_train = 5000
analyzer = SentimentIntensityAnalyzer()

all_tweets = get_tweets()
test_tweets = all_tweets[num_train:]

polarities = []
for tweet in test_tweets:
    ps = analyzer.polarity_scores(tweet.text)
    polarity = ps['compound']
    if polarity >= .33:
        polarities.append(3)
    elif polarity >= -.33:
        polarities.append(2)
    else:
        polarities.append(1)

print(polarities)
print(list(map(lambda tweet: tweet.sentiment, test_tweets)))
score = len(list(filter(lambda i: polarities[i] == test_tweets[i].sentiment, range(0, len(test_tweets))))) / len(test_tweets)
print("score: " + str(score))
