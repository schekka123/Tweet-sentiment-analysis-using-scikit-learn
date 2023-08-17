import string
from typing import List

import nltk
from nltk import WordNetLemmatizer

def get_lemmatized_text(tokenized_text: List[str]) -> List[str]:
    stop_words = []  # disabled temporarily - stopwords.words("english")
    stops = stop_words + [string.punctuation] + ["''", 'r.', '``', "'s", "n't"]
    filtered_words = list(map(lambda w: w.lower(), filter(lambda w: w.lower() not in stops, tokenized_text)))

    lemmatizer = WordNetLemmatizer()
    lemm_words = list(map(lambda w: lemmatizer.lemmatize(w), filtered_words))

    return lemm_words


def get_nouns_adjs_and_verb_tags(pos_tagged_text: list) -> list:
    # noun starts with N
    # adj starts with J
    # verb starts with V
    filtered_pos_tags = filter(
        lambda token_and_pos: token_and_pos[1].startswith("N") or token_and_pos[1].startswith("V") or token_and_pos[
            1].startswith("J"), pos_tagged_text)
    return list(filtered_pos_tags)


class SelfDrivingCarTweet:
    # sentiment from 1-5, -1 for not relevant
    def __init__(self, tweet_id: int, text: str, sentiment: int):
        self.id = tweet_id
        if sentiment in [1,2]:
            self.sentiment = 1
        elif sentiment == 3:
            self.sentiment = 2
        elif sentiment in [4,5]:
            self.sentiment = 3
        else:
            self.sentiment = sentiment
        self.text = text
        # get words
        self.tokenized_text = nltk.word_tokenize(text)
        # get LEMMATIZED words
        self.lemmatized_text = get_lemmatized_text(self.tokenized_text)
        # tag words so that we can extract nouns, adjs, and verbs
        self.pos_tagged_text = nltk.pos_tag(self.lemmatized_text)
        self.nouns_adjs_and_verb_tags = get_nouns_adjs_and_verb_tags(self.pos_tagged_text)
        self.nouns_adjs_and_verb_tokens = list(
            map(lambda token_and_pos: token_and_pos[0], self.nouns_adjs_and_verb_tags))
        self.nouns_adjs_and_verb_tokens_dict = dict()
        # create word -> part-of-speech dict
        for token_tuple in self.nouns_adjs_and_verb_tags:
            self.nouns_adjs_and_verb_tokens_dict[token_tuple[0]] = token_tuple[1]
