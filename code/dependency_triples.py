"""
Uses dependency triples
"""
import nltk
import numpy as np
from nltk.parse.malt import MaltParser
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import get_tweets


# convert the dependency triples into a comma-separated list
def transform_dependency_triples_to_tfidf_strings(triples):
    tfidf_strings = []
    for tweet in triples:
        def map_triple_to_string(triple):
            return "" if triple["word"] is None else triple["word"] + "," + "" if triple["head"] is None else \
                triple["head"] + "," + triple["tag"]

        tfidf_strings.append(" ".join(list(map(map_triple_to_string, tweet))))

    return tfidf_strings


def get_model_dependency_triples(mp, train_tweets: list):
    print("tweet length: " + str(len(train_tweets)))
    dependency_triples = parse_tweets_to_dependency_triples(mp, train_tweets)
    vectorizer = TfidfVectorizer()
    vectorizable_triples = transform_dependency_triples_to_tfidf_strings(dependency_triples)
    # print(vectorizable_triples)
    trained_X = vectorizer.fit_transform(vectorizable_triples)

    return vectorizer, trained_X


test_invalid_word = "NOTVALIDWORDDDDDDD"

# this converts the tweets into a list of dependency triples
# it then runs it through MaltParser to get a list of triples
# it then reconstructs the original tweets from the list output from mp
def parse_tweets_to_dependency_triples(mp, tweets: list):
    per_tweet_sentences = list(
        map(lambda tweet: list(map(lambda sent: sent.split(), nltk.sent_tokenize(tweet.text))), tweets))

    tweets_tokenized_sentences = []
    for tweet_sentences in per_tweet_sentences:
        for tweet_sentence in tweet_sentences:
            tweets_tokenized_sentences.append(tweet_sentence)
        tweets_tokenized_sentences.append([test_invalid_word])

    dependency_triples = []
    parsed_tweets = mp.parse_sents(tweets_tokenized_sentences)
    for parsed_sent_generator in parsed_tweets:
        for parsed_sent in parsed_sent_generator:
            parsed_nodes = parsed_sent.nodes
            triples = list(map(lambda node: {
                "word": node["word"],
                "head": parsed_nodes[node["head"]]["word"] if node["head"] is not None else np.nan,
                "tag": node["tag"]},
                               list(parsed_nodes.values())))
            for triple in triples:
                dependency_triples.append(triple)

    # print("deps:")
    # print(dependency_triples)

    # reconstruct tweets from sentence-based dependency triples
    current_tweet = 0
    dependency_triples_per_tweet = []
    current_tweet_triples = []
    for dependency_triple in dependency_triples:
        if dependency_triple["word"] == test_invalid_word:
            dependency_triples_per_tweet.append(current_tweet_triples)
            current_tweet_triples = []
            current_tweet = current_tweet + 1
        else:
            current_tweet_triples.append(dependency_triple)

    return dependency_triples_per_tweet


if __name__ == "__main__":
    mp = MaltParser('maltparser-1.8.1', 'engmalt.linear-1.7.mco')
    all_tweets = get_tweets()
    num_train = 5000
    train_tweets = all_tweets[0:num_train]
    test_tweets = all_tweets[num_train:]

    results = []
    model, trainedX = get_model_dependency_triples(mp, train_tweets)
    clf = svm.SVC()
    text_clf = clf.fit(trainedX, list(map(lambda tweet: tweet.sentiment, train_tweets)))
    predicted = text_clf.predict(model.transform(
        transform_dependency_triples_to_tfidf_strings(parse_tweets_to_dependency_triples(mp, test_tweets))))
    score = np.mean(predicted == list(map(lambda tweet: tweet.sentiment, test_tweets)))
    print("score: " + str(score))
