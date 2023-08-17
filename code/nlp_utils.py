from typing import List

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

import CarSentiment
import utils


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def get_nouns_adj_verb_strings_for_tweets(tweets: List[CarSentiment.SelfDrivingCarTweet]) -> List[str]:
    # concatenate the noun, adjs, and verbs into a sentence
    noun_adj_verb_strings = list(map(lambda tweet: " ".join(tweet.nouns_adjs_and_verb_tokens), tweets))
    return noun_adj_verb_strings


# tfidf approach instead of countvectorizer
def get_noun_adj_verb_bag_of_words_tfidf(tweets: List[CarSentiment.SelfDrivingCarTweet], **kwargs):
    tweet_noun_adj_verb_list: list = get_nouns_adj_verb_strings_for_tweets(tweets)
    bag_of_words_model = TfidfVectorizer(**kwargs)

    bag_of_words = bag_of_words_model.fit_transform(tweet_noun_adj_verb_list)
    # bag_of_words = pd.DataFrame(bag_of_words.toarray())
    # bag_of_words.columns = bag_of_words_model.get_feature_names_out()

    return bag_of_words_model, bag_of_words


# valid polarities: negative, positive, neutral, both
def get_sentiment_dict_for_tweet(tweet: CarSentiment.SelfDrivingCarTweet,
                                 subjectivity_lexicon: List[utils.SubjectivityWord],
                                 convert_to_number: bool = False):
    sentiment_dictionary = dict()
    for possible_polarity in utils.valid_polarities:
        sentiment_dictionary[possible_polarity] = 0

    for word in subjectivity_lexicon:
        if word.word in tweet.nouns_adjs_and_verb_tokens_dict.keys():
            polarity = word.polarity
            token_type = word.pos
            should_add = False
            if token_type == "anypos":
                should_add = True
            elif token_type == "noun" and tweet.nouns_adjs_and_verb_tokens_dict[word.word].startswith("N"):
                should_add = True
            elif token_type == "verb" and tweet.nouns_adjs_and_verb_tokens_dict[word.word].startswith("V"):
                should_add = True
            elif token_type == "adj" and tweet.nouns_adjs_and_verb_tokens_dict[word.word].startswith("J"):
                should_add = True
            if should_add:
                sentiment_dictionary[polarity] = sentiment_dictionary[polarity] + 1

    str_append = ["positive"] * sentiment_dictionary["positive"] + ["negative"] * sentiment_dictionary[
        "positive"] + ["neutral"] * sentiment_dictionary["neutral"]
    if not convert_to_number:
        return sentiment_dictionary, str_append
    else:
        return {"positivity_score": sentiment_dictionary["positive"] - sentiment_dictionary["negative"]}, str_append
