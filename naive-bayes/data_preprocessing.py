
import pandas as pd
import numpy as np
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer



class TextPreprocesser:
    def __init__(self, csv_path: str="", save_to: str=""):
        self.dataframe = pd.read_csv(csv_path)
        self.emails = self.dataframe["v2"].tolist()
        self.labels = self.dataframe["v1"].tolist()

        self.tokenizer = RegexpTokenizer(r"\w+")
        self.stop_words = set(stopwords.words("english"))
        self.ps = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        self.save_to = save_to

    """ convert sentence to list """
    def _tokenize(self, email: str) -> list:
        return self.tokenizer.tokenize(email)

    """ remove stop-words """
    def _rm_stop_words(self, email: list) -> list:
        return [word.lower() for word in email if not word in self.stop_words]

    """ stem words """
    def _stem(self, email: list) -> list:
        return [self.ps.stem(word) for word in email]

    """ lemmetize nouns """ 
    def _lemmatize(self, email: list) -> list:
        return [self.lemmatizer.lemmatize(word) for word in email]

    """ apply nltk preprocessing steps on one email """
    def _apply(self, email: str):
        email = self._tokenize(email)
        email = self._rm_stop_words(email)
        email = self._stem(email)
        email = self._lemmatize(email)
        return email

    """ split dataset into train and test batch """
    def _split(self, samples: list, labels: list, test_size: float):
        split_index = int(len(samples) * test_size)
        train_x, test_x = samples[split_index:], samples[:split_index]
        train_y, test_y = labels[split_index:], labels[:split_index]

        return train_x, test_x, train_y, test_y

    """ preprocess dataset """
    def preprocess_dataset(self, test_size: float=0.2):
        samples, labels = [], []
        for idx in range(len(self.emails)):
            samples.append(self._apply(self.emails[idx]))
            labels.append(1) if self.labels[idx] == "spam" else labels.append(0)

        train_x, test_x, train_y, test_y = self._split(samples, labels, test_size)

        with open(self.save_to, "w+") as f:
            json.dump([train_x, test_x, train_y, test_y], f)
        

if __name__ == "__main__":
    textPreprocesser = TextPreprocesser(csv_path="data/spam-and-ham/spam.csv", save_to="data/spam-and-ham/dataset.json")
    textPreprocesser.preprocess_dataset(test_size=0.1)