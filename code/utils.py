import re
from csv import reader
from typing import List

from matplotlib import pyplot as plt

import CarSentiment


def plot(title, x_ys):
    x = list(map(lambda tuple: tuple[0], x_ys))
    y = list(map(lambda tuple: tuple[1], x_ys))

    x_pos = [i for i, _ in enumerate(x)]

    plt.bar(x_pos, y)
    plt.xlabel("Tweet Categories")
    plt.ylabel("Score")
    plt.title(title)

    plt.xticks(x_pos, x)
    plt.ylim((0, 1))
    plt.show()


def get_tweets() -> List[CarSentiment.SelfDrivingCarTweet]:
    with open("Twitter-sentiment-self-drive-DFE.csv", 'r') as csv:
        # print(csv)
        csv_reader = reader(csv)
        header = next(csv_reader)
        if header is not None:
            tweets: List[CarSentiment.SelfDrivingCarTweet] = []
            for row in csv_reader:
                # print(row)
                tweet_id = int(row[0])
                sentiment = int(row[5]) if row[5] != "not_relevant" else -1
                text = row[10]
                split_text = text.split(" ")
                split_text = filter(lambda word: len(word) > 1 and not word.startswith("http"), split_text)
                text = " ".join(split_text)
                tweets.append(CarSentiment.SelfDrivingCarTweet(tweet_id, text, sentiment))
            return tweets
        else:
            raise ValueError()


class SubjectivityWord:
    def __init__(self, type: str, len: int, word: str, pos: str, stemmed: bool, polarity: str):
        self.type = type
        self.len = len
        self.word = word
        self.pos = pos
        self.stemmed = stemmed
        self.polarity = polarity


# valid polarities:
# negative, positive, neutral, both
# return a list of words by their polarity given the lexicon file
valid_polarities = ["negative", "positive", "neutral", "both"]
def get_subjectivity_lexicon() -> List[SubjectivityWord]:
    subjectivity_regex = r"type=(.+)\slen=(\d+)\sword1=(.+)\spos1=(adj|verb|noun|anypos)\sstemmed1=(y|n)\spriorpolarity=(positive|negative|both|neutral)\n"
    subjectivity_lexicon_text: str = open("subjclueslen1-HLTEMNLP05.tff", "r").read()
    subjectivity_lexicon: List[SubjectivityWord] = []
    for match in re.finditer(subjectivity_regex, subjectivity_lexicon_text):
        groups = list(match.groups())
        subjectivity_lexicon.append(
            SubjectivityWord(groups[0], int(groups[1]), groups[2], groups[3], groups[4] == "y", groups[5]))
    return subjectivity_lexicon
