from typing import List

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

import CarSentiment
import nlp_utils
from utils import get_tweets, get_subjectivity_lexicon

num_train = 5500

all_tweets = get_tweets()
train_tweets = all_tweets[0:num_train]
test_tweets = all_tweets[num_train:]

subjectivity_lexicon = get_subjectivity_lexicon()


def get_feature_dicts_for_tweets(tweets: List[CarSentiment.SelfDrivingCarTweet],
                                 convert_to_number: bool = False):
    feature_dicts_list = []
    sentiment_list = []
    for tweet in tweets:
        sentiment_dict, str_append = nlp_utils.get_sentiment_dict_for_tweet(tweet, subjectivity_lexicon,
                                                                            convert_to_number)
        feature_dicts_list.append(" ".join(tweet.lemmatized_text + str_append))
        sentiment_list.append(tweet.sentiment)
    return feature_dicts_list, sentiment_list


train_X, train_Y = get_feature_dicts_for_tweets(train_tweets, convert_to_number=True)
test_X, test_Y = get_feature_dicts_for_tweets(test_tweets, convert_to_number=True)

pipeline = make_pipeline(TfidfVectorizer(), svm.SVC())
pipeline.fit(train_X, train_Y)

predicted = pipeline.predict(test_X)
score = len(list(filter(lambda i: test_Y[i] == predicted[i], range(0, len(predicted))))) / len(predicted)
print("score: " + str(score))
