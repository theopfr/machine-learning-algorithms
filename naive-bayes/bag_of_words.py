
import numpy as np
from tqdm import tqdm
import json


""" split dataset into classes and log every word """
def split_classes(emails: list, labels: list) -> list:
    spam, ham = [], []
    total_words = {}
    for idx in range(len(emails)):
        if labels[idx] == 1: spam.append(emails[idx])
        elif labels[idx] == 0: ham.append(emails[idx])

        for word in emails[idx]:
            total_words[word] = {"p(w)": 0, "p(w|spam)": 0, "p(w|ham)": 0}

    return spam, ham, total_words

""" save bag of words to json """
def save_bag(data: dict, save_to: str):
    with open(save_to, "w+") as f:
        json.dump(data, f)

""" create bag of words containung probabilites """
def create_word_probability_bag(train_samples: list=[], train_labels: list=[], save_to: str=""):
    emails, labels = train_samples, train_labels
    spam, ham, total_words = split_classes(emails, labels)

    total_messages_count_spam = len(spam)
    total_messages_count_ham = len(ham)
    total_messages_count = len(spam) + len(ham)

    total_word_count_spam = sum(len(email) for email in spam)
    total_word_count_ham = sum(len(email) for email in ham)
    total_word_count = total_word_count_spam + total_word_count_ham

    bag_of_words = {"total_messages_count_spam": total_messages_count_spam, 
                    "total_messages_count_ham": total_messages_count_ham,
                    "total_messages_count": total_messages_count,
                    "total_word_count_spam": total_word_count_spam,
                    "total_word_count_ham": total_word_count_ham,
                    "total_word_count": total_word_count}

    for current_word in tqdm(total_words, desc="creating bag of words"):
        occurences_total = 0
        occurences_spam = 0
        occurences_ham = 0
        total_messages_containing_word = 0

        for email in spam:
            email_contains_word = False
            for word in email:
                if word == current_word:
                    occurences_spam += 1
                    email_contains_word = True
            
            if email_contains_word:
                total_messages_containing_word += 1

        for email in ham:
            email_contains_word = False
            for word in email:
                if word == current_word:
                    occurences_ham += 1
                    email_contains_word = True

            if email_contains_word:
                total_messages_containing_word += 1

        occurences_total = occurences_spam + occurences_ham

        total_words[current_word]["p(w)"] = occurences_total / total_word_count

        # if word is not in dataset, naive suppose it appears 0.1 times
        pw_spam = occurences_spam / total_word_count_spam
        total_words[current_word]["p(w|spam)"] = pw_spam if pw_spam > 0.0 else 0.1 / total_messages_count_spam

        pw_ham = occurences_ham / total_word_count_ham
        total_words[current_word]["p(w|ham)"] = pw_ham if pw_ham > 0.0 else 0.1 / total_messages_count_ham

    bag_of_words["bag"] = total_words
    save_bag(bag_of_words, save_to)

    



