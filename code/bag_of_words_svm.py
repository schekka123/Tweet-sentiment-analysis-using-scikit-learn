"""
This uses a bag of words, testing over just unigrams, both unigrams and bigrams,
and then unigrams, bigrams, and trigrams, to create feature vectors.

This approach is void of all context
"""
import numpy as np
from sklearn import svm

from nlp_utils import get_nouns_adj_verb_strings_for_tweets, get_noun_adj_verb_bag_of_words_tfidf
from utils import get_tweets
from utils import plot

num_train = 5500

all_tweets = get_tweets()
train_tweets = all_tweets[0:num_train]
test_tweets = all_tweets[num_train:]

ngram_ranges = [[1, 1], [1, 2], [1, 3]]
results = []
for ngram_range in ngram_ranges:
    # get model
    tweets_bag_of_words_model, tweets_bag_of_words_train = get_noun_adj_verb_bag_of_words_tfidf(train_tweets,
                                                                                                ngram_range=ngram_range)
    tweets_bag_of_words_test = tweets_bag_of_words_model.transform(get_nouns_adj_verb_strings_for_tweets(test_tweets))
    svc = svm.SVC()
    clf = svc
    text_clf = clf.fit(tweets_bag_of_words_train, list(map(lambda tweet: tweet.sentiment, train_tweets)))
    # print("best svc params: " + str(clf.best_params_))
    predicted = text_clf.predict(tweets_bag_of_words_test)
    score = np.mean(predicted == list(map(lambda tweet: tweet.sentiment, test_tweets)))
    print("score for " + str(ngram_range) + ": " + str(score))
    results.append([str(ngram_range), score])

plot("Using bag of words", results)
