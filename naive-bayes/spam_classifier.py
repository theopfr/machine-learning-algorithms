
from functools import reduce
import json
import numpy as np
import os
from bag_of_words import create_word_probability_bag


class NaiveBayes:
    def __init__(self, dataset_path: str="", bag_of_words_path: str=""):
        with open(dataset_path, "r") as f: 
            dataset = json.load(f)
        self.train_x, self.test_x, self.train_y, self.test_y = dataset[0], dataset[1], dataset[2], dataset[3], 
        self.bag_of_words_path = bag_of_words_path

    """ create bag of words and save to json """
    def _get_bag_of_words(self, create_new_bag: bool=False):
        if os.stat(self.bag_of_words_path).st_size == 0 or create_new_bag:

            create_word_probability_bag(train_samples=self.train_x, 
                                        train_labels=self.train_y, 
                                        save_to=self.bag_of_words_path)
        
        with open(self.bag_of_words_path, "r") as f:
            return json.load(f)

    """ classify email """
    def _classify(self, bag_of_words: dict, email: list):
        total_messages_count_spam = bag_of_words["total_messages_count_spam"]
        total_messages_count_ham = bag_of_words["total_messages_count_ham"] 
        total_messages_count = bag_of_words["total_messages_count"]
        total_word_count_spam = bag_of_words["total_word_count_spam"] 
        total_word_count_ham = bag_of_words["total_word_count_ham"] 
        total_word_count = bag_of_words["total_word_count"]

        word_bag = bag_of_words["bag"]

        p_spam = total_messages_count_spam / total_messages_count
        p_ham = total_messages_count_ham / total_messages_count

        word_spam_probablities = []
        word_ham_probablities = []
        for word in email:
            try:
                word_spam_probablities.append(word_bag[word]["p(w|spam)"])
                word_ham_probablities.append(word_bag[word]["p(w|ham)"])
            except:
                word_spam_probablities.append(1 / total_messages_count_spam)
                word_ham_probablities.append(1 / total_messages_count_ham)

        v_spam = reduce(lambda x, y: x*y, word_spam_probablities) * p_spam
        v_ham = reduce(lambda x, y: x*y, word_ham_probablities) * p_ham

        is_spam = v_spam / (v_spam + v_ham)
        is_ham = v_ham / (v_ham + v_spam)

        return is_spam, is_ham

    """ test model """
    def test(self):
        correct = 0

        bag_of_words = self._get_bag_of_words(create_new_bag=True)
        for idx in range(len(self.test_x)):
            email, label = self.test_x[idx], self.test_y[idx]

            probablity_spam, probablity_ham = self._classify(bag_of_words, email)
            
            if round(probablity_spam, 0) == label:
                correct += 1

        print("accuracy:", 100 * round(correct / len(self.test_x), 4), "%")


naiveBayes = NaiveBayes(dataset_path="data/spam-and-ham/dataset.json", bag_of_words_path="data/bag_of_words.json")
naiveBayes.test()

    
