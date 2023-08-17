from typing import List

from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline

import CarSentiment
import nlp_utils
from utils import get_tweets, get_subjectivity_lexicon

num_train = 5500

all_tweets = get_tweets()
train_tweets = all_tweets[0:num_train]
test_tweets = all_tweets[num_train:]

subjectivity_lexicon = get_subjectivity_lexicon()


def get_feature_dicts_for_tweets(tweets: List[CarSentiment.SelfDrivingCarTweet]) -> List[tuple]:
    feature_dicts = []
    for tweet in tweets:
        sentiment_dict, str_append = nlp_utils.get_sentiment_dict_for_tweet(tweet, subjectivity_lexicon)
        sentiment_dict["length"] = len(tweet.text)  # see if tweet raw length matters
        feature_dicts.append((sentiment_dict, tweet.sentiment))

    return feature_dicts


train_feature_dicts = get_feature_dicts_for_tweets(train_tweets)
test_feature_dicts = get_feature_dicts_for_tweets(test_tweets)
train_X = list(map(lambda tup: tup[0], train_feature_dicts))
train_Y = list(map(lambda tup: tup[1], train_feature_dicts))
pipeline = make_pipeline(DictVectorizer(sparse=True), svm.SVC()) # dictionary of features, so dictvectorizer
pipeline.fit(train_X, train_Y)

test_X = list(map(lambda tup: tup[0], test_feature_dicts))
test_Y = list(map(lambda tup: tup[1], test_feature_dicts))

predicted = pipeline.predict(test_X)
score = len(list(filter(lambda i: test_Y[i] == predicted[i], range(0, len(predicted))))) / len(predicted)
print("score: " + str(score))
