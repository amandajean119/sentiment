from __future__ import division
import cPickle
import os
import sys
import time
from itertools import izip
from math import log
from os import system
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pylab
from os.path import isfile, join


class MyDict(dict):
    def __getitem__(self, key):
        if key in self:
            return self.get(key)
        return 0


class NBSC:
    """

    """

    def __init__(self, filepath):
        """

        :rtype: NBSC object
        """
        self.filepath = filepath
        self.very_pos = MyDict()
        self.pos = MyDict()
        self.neutral = MyDict()
        self.neg = MyDict()
        self.very_neg = MyDict()
        self.totals = [0, 0, 0, 0, 0]
        # self.delchars = ''.join(c for c in map(chr, range(128)) if not c.isalnum())
        self.match_num_score = {'1': 'very_neg', '2': 'neg', '3': 'neutral', '4': 'pos', '5': 'very_pos'}
        self.features = set()

    @staticmethod
    def negate_sequence(words):
        """
        Detects negations and transforms negated words into "not_" form.
        :type words: str
        :param words:
        :return:
        """
        negation = False
        delims = "?.,!:;"
        result = []
        words = words.split()
        prev = None
        pprev = None
        for word in words:
            # stripped = word.strip(delchars)
            stripped = word.strip(delims).lower()
            negated = "not_" + stripped if negation else stripped
            result.append(negated)
            if prev:
                bigram = prev + " " + negated
                result.append(bigram)
                if pprev:
                    trigram = pprev + " " + bigram
                    result.append(trigram)
                pprev = prev
            prev = negated

            if any(neg in word for neg in ["not", "n't", "no"]):
                negation = not negation

            if any(c in word for c in delims):
                negation = False

        return result

    def train(self, train_file, meta_file, new_model_file):
        """
        Performs training of a Naive Bayes Sentiment Classifier
        :param train_file: File containing <review> per line
        :param meta_file: File containing <ID>,<review score>
        :param new_model_file: Filename of new model (pickle)
        :return: None
        """
        t = time.time()
        with open(train_file, 'r') as f:
            with open(meta_file, 'r') as meta_file:
                for review_text, meta in izip(f, meta_file):
                    meta = meta.split(',')
                    score = int(float(meta[-1].strip()))
                    for word in set(NBSC.negate_sequence(review_text)):
                        if score == 5:
                            self.very_pos[word] += 1
                        elif score == 4:
                            self.pos[word] += 1
                        elif score == 3:
                            self.neutral[word] += 1
                        elif score == 2:
                            self.neg[word] += 1
                        elif score == 1:
                            self.very_neg[word] += 1
                        else:
                            print "Bad score"
                            print score
        print "Accumulated features"
        self.prune_features()
        print "Pruned features"
        # Next step would be to take mutual information into account, either by a threshold or max num of features
        self.totals[0] = sum(self.very_pos.values())
        self.totals[1] = sum(self.pos.values())
        self.totals[2] = sum(self.neutral.values())
        self.totals[3] = sum(self.neg.values())
        self.totals[4] = sum(self.very_neg.values())
        print self.totals
        new_model = (self.very_pos, self.pos, self.neutral, self.neg, self.very_neg, self.totals)
        cPickle.dump(new_model, open(new_model_file, 'w'))
        elapsed = time.time() - t
        print elapsed

    def prune_features(self):
        """
        Remove features that appear only once.
        """
        for k in self.very_pos.keys():
            if self.very_pos[k] <= 1 and self.pos[k] <= 1 and self.neutral[k] <= 1 and self.neg[k] <= 1 and \
                            self.very_neg[k] <= 1:
                del self.very_pos[k]
        for k in self.pos.keys():
            if self.very_pos[k] <= 1 and self.pos[k] <= 1 and self.neutral[k] <= 1 and self.neg[k] <= 1 and \
                            self.very_neg[k] <= 1:
                del self.pos[k]
        for k in self.neutral.keys():
            if self.very_pos[k] <= 1 and self.pos[k] <= 1 and self.neutral[k] <= 1 and self.neg[k] <= 1 and \
                            self.very_neg[k] <= 1:
                del self.neutral[k]
        for k in self.neg.keys():
            if self.very_pos[k] <= 1 and self.pos[k] <= 1 and self.neutral[k] <= 1 and self.neg[k] <= 1 and \
                            self.very_neg[k] <= 1:
                del self.neg[k]
        for k in self.very_neg.keys():
            if self.very_pos[k] <= 1 and self.pos[k] <= 1 and self.neutral[k] <= 1 and self.neg[k] <= 1 and \
                            self.very_neg[k] <= 1:
                del self.very_neg[k]

    def MI(self, word):
        """
        Compute the weighted mutual information of a term.
        :param word: word we want to find the weighted mutual information of
        :return: the weighted mutual information of the word
        """
        T = sum(self.totals)
        W = self.very_pos[word] + self.pos[word] + self.neutral[word] + self.neg[word] + self.very_neg[word]
        I = 0
        if W == 0:
            return 0

        if self.very_pos[word] > 0:
            # doesn't occur in +ve
            I += (self.totals[0] - self.very_pos[word]) / T * log ((self.totals[0] - self.very_pos[word]) * T / (T - W) / self.totals[0])
            # occurs in +ve
            I += self.very_pos[word] / T * log(self.very_pos[word] * T / W / self.totals[0])
        if self.pos[word] > 0:
            # doesn't occur in +ve
            I += (self.totals[1] - self.pos[word]) / T * log ((self.totals[1] - self.pos[word]) * T / (T - W) / self.totals[1])
            # occurs in +ve
            I += self.pos[word] / T * log (self.pos[word] * T / W / self.totals[1])
        if self.neutral[word] > 0:
            # doesn't occur in +ve
            I += (self.totals[2] - self.neutral[word]) / T * log ((self.totals[2] - self.neutral[word]) * T / (T - W) / self.totals[2])
            # occurs in +ve
            I += self.neutral[word] / T * log (self.neutral[word] * T / W / self.totals[2])
        if self.neg[word] > 0:
            # doesn't occur in -ve
            I += (self.totals[3] - self.neg[word]) / T * log ((self.totals[3] - self.neg[word]) * T / (T - W) / self.totals[3])
            # occurs in -ve
            I += self.neg[word] / T * log (self.neg[word] * T / W / self.totals[3])
        if self.very_neg[word] > 0:
            # doesn't occur in -ve
            I += (self.totals[4] - self.very_neg[word]) / T * log ((self.totals[4] - self.very_neg[word]) * T / (T - W) / self.totals[4])
            # occurs in -ve
            I += self.very_neg[word] / T * log (self.very_neg[word] * T / W / self.totals[4])

        return I

    def classify(self, words):
        """
        For classification from pretrained data
        :param words: Review we want to classify
        :return: List of probabilities for each sentiment class
        """
        words = set(word for word in NBSC.negate_sequence(words) if word in self.very_pos or word in self.pos or
                    word in self.neutral or word in self.neg or word in self.very_neg)
        if len(words) == 0:
            return [0, 0, 0, 0, 0]
        # Probability that word occurs in pos documents
        if self.totals[0] != 0:
            very_pos_prob = sum(log(float((self.very_pos[word] + 1)) / (2.0 * float(self.totals[0]))) for word in words)
        else:
            very_pos_prob = 0
        if self.totals[1] != 0:
            pos_prob = sum(log(float((self.pos[word] + 1)) / (2.0 * float(self.totals[1]))) for word in words)
        else:
            pos_prob = 0
        if self.totals[2] != 0:
            neutral_prob = sum(log(float((self.neutral[word] + 1)) / (2.0 * float(self.totals[2]))) for word in words)
        else:
            neutral_prob = 0
        if self.totals[3] != 0:
            neg_prob = sum(log(float((self.neg[word] + 1)) / (2.0 * float(self.totals[3]))) for word in words)
        else:
            neg_prob = 0
        if self.totals[4] != 0:
            very_neg_prob = sum(log(float((self.very_neg[word] + 1)) / (2.0 * float(self.totals[4]))) for word in words)
        else:
            very_neg_prob = 0
        scores = [very_pos_prob, pos_prob, neutral_prob, neg_prob, very_neg_prob]
        return scores

    def get_relevant_features(self):
        """
        Only keep words that are in a features list
        :return: Model containing a subset of words
        """
        very_pos_dump = MyDict({k: self.very_pos[k] for k in self.very_pos if k in self.features})
        pos_dump = MyDict({k: self.pos[k] for k in self.pos if k in self.features})
        neutral_dump = MyDict({k: self.neutral[k] for k in self.neutral if k in self.features})
        neg_dump = MyDict({k: self.neg[k] for k in self.neg if k in self.features})
        very_neg_dump = MyDict({k: self.very_neg[k] for k in self.very_neg if k in self.features})
        totals_dump = [sum(very_pos_dump.values()), sum(pos_dump.values()), sum(neutral_dump.values()), sum(neg_dump.values()), sum(very_neg_dump.values())]
        #Need to free memory?
        return (very_pos_dump, pos_dump, neutral_dump, neg_dump, very_neg_dump, totals_dump)

    def feature_selection_trials(self):
        """
        Select top k features. Vary k and plot data
        :return: None
        """
        model_file = self.filepath + 'models/model_round3.pickle'
        (self.very_pos, self.pos, self.neutral, self.neg, self.very_neg, self.totals) = cPickle.load(open(model_file))
        words = list(set(self.very_pos.keys() + self.pos.keys() + self.neutral.keys() + self.neg.keys() + self.very_neg.keys()))
        print "Total no of features:", len(words)
        words.sort(key=lambda w: -self.MI(w))
        num_features, accuracy = [], []
        bestk = 0
        limit = 500
        output_path = self.filepath + 'feature_experiments/'
        step = 10000
        start = 50000
        best_accuracy = 0.0
        round_number = "round1"
        for w in words[:start]:
            self.features.add(w)
        for k in xrange(start, len(words), step):
            for w in words[k:k+step]:
                self.features.add(w)
            correct = 0
            size = 0
            input_path = self.filepath + round_number + '/' + round_number + 'input.txt'
            meta_file = self.filepath + round_number + '/' + round_number + 'meta.txt'
            with open(input_path, 'r') as input_file:
                with open(meta_file, 'r') as m:
                    for (line, meta) in izip(input_file, m):
                        probs = self.classify(line.strip())
                        review_class = np.argmax(probs) + 1
                        meta = meta.split(',')
                        if review_class == int(float(meta[1])):
                            correct += 1
                        size += 1
            num_features.append(k+step)
            accuracy.append(correct / size)
            if (correct / size) > best_accuracy:
                bestk = k
            print k+step, correct / size

        self.features = set(words[:bestk])
        cPickle.dump(self.get_relevant_features(), open(output_path + 'best_model.pickle', 'w'))

        pylab.plot(num_features, accuracy)
        pylab.show()

    def get_stanford_data(self):
        """
        Get rottentomatoes.com Stanford data from sentence fragment files (http://nlp.stanford.edu/sentiment/index.html)
        :return: Model trained from scratch on this dataset
        """
        phrases = pd.read_csv(self.filepath + 'stanfordSentimentTreebank/dictionary.txt',
                              sep='|', header=None, names=['phrase', 'id'])
        phrase_scores = pd.read_csv(self.filepath + 'stanfordSentimentTreebank/sentiment_labels.txt',
                                    sep='|', header=None, names=['id', 'score'])
        merged_phrases = pd.merge(phrases, phrase_scores, how='inner', on='id')
        merged_phrases['fixed_score'] = 0
        merged_phrases.loc[merged_phrases.score <= 0.2, 'fixed_score'] = 1
        merged_phrases.loc[(merged_phrases['score'] > 0.2) & (merged_phrases['score'] <= 0.4), 'fixed_score'] = 2
        merged_phrases.loc[(merged_phrases['score'] > 0.4) & (merged_phrases['score'] <= 0.6), 'fixed_score'] = 3
        merged_phrases.loc[(merged_phrases['score'] > 0.6) & (merged_phrases['score'] <= 0.8), 'fixed_score'] = 4
        merged_phrases.loc[(merged_phrases['score'] > 0.8) & (merged_phrases['score'] <= 1.0), 'fixed_score'] = 5
        print merged_phrases[merged_phrases['fixed_score'] == 1].shape[0]
        print merged_phrases[merged_phrases['fixed_score'] == 2].shape[0]
        print merged_phrases[merged_phrases['fixed_score'] == 3].shape[0]
        print merged_phrases[merged_phrases['fixed_score'] == 4].shape[0]
        print merged_phrases[merged_phrases['fixed_score'] == 5].shape[0]
        merged_phrases.to_csv(self.filepath + 'stanford_labeled_phrases.csv',
                              sep='|', index=False, columns=['phrase', 'fixed_score'])
        training_vals = merged_phrases[['phrase', 'fixed_score']].values
        for row in training_vals:
            for word in set(NBSC.negate_sequence(row[0])):
                if word:
                    score = row[1]
                    if score == 5:
                        self.very_pos[word] += 1
                    elif score == 4:
                        self.pos[word] += 1
                    elif score == 3:
                        self.neutral[word] += 1
                    elif score == 2:
                        self.neg[word] += 1
                    elif score == 1:
                        self.very_neg[word] += 1
                    else:
                        print "Bad score"
                        print score
        print "Accumulated Stanford features"

        self.prune_features()
        print "Pruned Stanford features"

        self.totals[0] = sum(self.very_pos.values())
        self.totals[1] = sum(self.pos.values())
        self.totals[2] = sum(self.neutral.values())
        self.totals[3] = sum(self.neg.values())
        self.totals[4] = sum(self.very_neg.values())
        print self.totals

        new_model = (self.very_pos, self.pos, self.neutral, self.neg, self.very_neg, self.totals)
        cPickle.dump(new_model, open(self.filepath + 'models/model_round0.pickle', 'w'))
        return new_model

    def get_training_and_testing(self, prev_round, curr_round):
        """
        Create next round training set based on agreement between test set sentiment classification score and label
        :param prev_round: Previous round
        :param curr_round: Next round
        :return: None
        """
        output_file = self.filepath + prev_round + '/' + prev_round + 'output.txt'
        training_file = self.filepath + prev_round + '/' + prev_round + 'training.txt'
        training_meta_file = self.filepath + prev_round + '/' + prev_round + 'training_meta.txt'
        testing_file = self.filepath + curr_round + '/' + curr_round + 'input.txt'
        testing_meta_file = self.filepath + curr_round + '/' + curr_round + 'meta.txt'
        matched_file = self.filepath + prev_round + '/' + prev_round + 'matched.txt'
        with open(output_file, 'r') as o:
            with open(training_file, 'w') as train_output:
                with open(training_meta_file, 'w') as train_meta_output:
                    with open(testing_file, 'w') as test_output:
                        with open(testing_meta_file, 'w') as test_meta_output:
                            with open(matched_file, 'w') as matched:
                                for line in o:
                                    line = line.strip().split(';')
                                    review_id = line[0]
                                    rating = int(float(line[1]))
                                    score = int(line[2])
                                    review = ';'.join(line[3:])
                                    '''
                                    if score == rating:
                                        train_output.write(review + '\n')
                                        train_meta_output.write(review_id + ',' + str(rating) + '\n')
                                    else:
                                        test_output.write(review + '\n')
                                        test_meta_output.write(review_id + ',' + str(rating) + '\n')
                                    matched.write(
                                        review_id + ';' + str(rating) + ';' + str(score) + ';' + review + '\n')
                                    '''
                                    if abs(score - rating) <=1:
                                        train_output.write(review + '\n')
                                        train_meta_output.write(review_id + ',' + str(rating) + '\n')
                                    else:
                                        test_output.write(review + '\n')
                                        test_meta_output.write(review_id + ',' + str(rating) + '\n')
                                    matched.write(
                                        review_id + ';' + str(rating) + ';' + str(score) + ';' + review + '\n')


    def round0(self):
        """
        Classify all test data based on initial model
        :return: None
        """
        (self.very_pos, self.pos, self.neutral, self.neg, self.very_neg, self.totals) = self.get_stanford_data()
        #model_file = self.filepath + 'models/model_round0.pickle'
        (self.very_pos, self.pos, self.neutral, self.neg, self.very_neg, self.totals) = cPickle.load(open(model_file))
        input_file = self.filepath + 'round1/round1input.txt'
        meta_file = self.filepath + 'round1/round1meta.txt'
        probs_file = self.filepath + 'round1/round1probs.txt'
        output_file = self.filepath + 'round1/round1output.txt'
        with open(input_file, 'r') as i:
            with open(meta_file, 'r') as m:
                with open(probs_file, 'w') as p:
                    with open(output_file, 'w') as o:
                        for (line, meta) in izip(i, m):
                            probs = self.classify(line.strip())
                            for item in meta.strip().split(','):
                                o.write(item + ';')
                                p.write(item + ';')
                            for prob in probs:
                                p.write(str(prob) + ';')
                            p.write(line.strip() + '\n')
                            prob = np.argmax(probs) + 1
                            o.write(str(prob) + ';')
                            o.write(line.strip() + '\n')

    def run_experiment_iterations(self):
        """
        Run 20 rounds of iterative training and testing, based on the paper at cs.unm.edu/~aminnich/clearview
        :return: None
        """
        i = 1
        while i < 21:
            prev_round = 'round' + str(i)
            i += 1
            round_number = 'round' + str(i)
            print round_number
            system('mkdir ' + self.filepath + round_number)

            self.get_training_and_testing(prev_round, round_number)
            print "Got testing and training"

            # once we generate the testing and training files, need to run the training instance
            new_model = self.filepath + "models/model_" + prev_round + ".pickle"
            train_file = self.filepath + prev_round + '/' + prev_round + 'training.txt'
            meta_file = self.filepath + prev_round + '/' + prev_round + 'meta.txt'
            self.train(train_file, meta_file, new_model)
            print "trained new model"

            # then we run the classifier on the newly trained model
            input_file = self.filepath + round_number + '/' + round_number + 'input.txt'
            meta_file = self.filepath + round_number + '/' + round_number + 'meta.txt'
            probs_file = self.filepath + round_number + '/' + round_number + 'probs.txt'
            output_file = self.filepath + round_number + '/' + round_number + 'output.txt'
            with open(input_file, 'r') as input:
                with open(meta_file, 'r') as m:
                    with open(probs_file, 'w') as p:
                        with open(output_file, 'w') as o:
                            for (line, meta) in izip(input, m):
                                probs = self.classify(line.strip())
                                for item in meta.strip().split(','):
                                    o.write(item + ';')
                                    p.write(item + ';')
                                for prob in probs:
                                    p.write(str(prob) + ';')
                                p.write(line.strip() + '\n')
                                prob = np.argmax(probs) + 1
                                o.write(str(prob) + ';')
                                o.write(line.strip() + '\n')
            print 'Classified testing set'

    def analyzeResults(self, experiment_type):
        """
        Produce output plots
        :param experiment_type: Level of agreement required to move a sample into training set.
        Specified so you can compare results from different runs.
        :return: None
        """
        ind = np.arange(20)
        width = 0.35
        amt = pd.ExcelFile(self.filepath + 'userStudyResults.xlsx')
        amt = amt.parse("userStudyResults.csv")
        image_path = self.filepath + 'images/' + experiment_type + '/'
        system('mkdir ' + image_path)
        round_number = "round1"
        output_file = self.filepath + round_number + '/' + round_number + 'matched.txt'
        review_ids = []
        ratings = []
        round_numbers = []
        # texts = []
        with open(output_file, 'r') as f:
            for line in f:
                line = line.split(';')
                review_ids.append(line[0])
                ratings.append(int(line[1]))
                round_numbers.append(int(line[2]))
        all_round_results = pd.DataFrame(zip(review_ids, ratings, round_numbers),
                                         columns=['review_id', 'rating', round_number])
        amt = pd.merge(amt, all_round_results, how='left', on='review_id')

        for i in range(2, 21):
            round_number = 'round' + str(i)
            output_file = self.filepath + round_number + '/' + round_number + 'matched.txt'
            # Take in the data this way because the review text contains semi-colons and Pandas
            # can't handle that. Can use commented code below if that is not the case for your data.
            review_ids = []
            round_numbers = []
            with open(output_file, 'r') as f:
                for line in f:
                    line = line.split(';')
                    review_ids.append(line[0])
                    round_numbers.append(int(line[2]))
            all_round_results = pd.DataFrame(zip(review_ids, round_numbers), columns=['review_id', round_number])
            #all_round_results = pd.read_csv(output_file, sep=';', header=None, names=['review_id', 'rating', round_number, 'text', 'ignore', 'ignore2'], usecols=['review_id', round_number])
            # data[round_number] = [float(x.strip()) for x in data[round_number].values]
            amt = pd.merge(amt, all_round_results, how='left', on='review_id')
            print round_number
            print amt[round_number].isnull().tolist().count(True)
        amt = amt.dropna(subset=['round1'], how='any')
        amt = amt.fillna(method='pad', axis=1)
        amt.to_csv(self.filepath + 'amt.csv')

        amt_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            amt_sc.append(amt[abs(amt['avg_reviewer_score'] - amt[round_number]) < 1].shape[0])
        print amt_sc
        f1 = plt.figure()
        ax = f1.add_subplot(111)
        ax.bar(ind, amt_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 10,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of Subset Samples Where User Study Results - Sentiment Classifier is < 1')
        ax.set_ylim((0, 10000))
        plt.savefig(image_path + 'subset_user_sent_0.png', bbox_inches='tight')

        amt_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            amt_sc.append(amt[abs(amt['avg_reviewer_score'] - amt[round_number]) <= 1].shape[0])
        print amt_sc
        f2 = plt.figure()
        ax = f2.add_subplot(111)
        ax.bar(ind, amt_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 10,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of Subset Samples Where User Study Results - Sentiment Classifier is <= 1')
        ax.set_ylim((0, 10000))
        plt.savefig(image_path + 'subset_user_sent_1.png', bbox_inches='tight')

        amt_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            amt_sc.append(amt[abs(amt['avg_reviewer_score'] - amt[round_number]) <= 2].shape[0])
        print amt_sc
        f3 = plt.figure()
        ax = f3.add_subplot(111)
        ax.bar(ind, amt_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 10,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of Subset Samples Where User Study Results - Sentiment Classifier is <= 2')
        ax.set_ylim((0, 10000))
        plt.savefig(image_path + 'subset_user_sent_2.png', bbox_inches='tight')

        twisters = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            twisters.append(amt[abs(amt['rating'] - amt[round_number]) >= 2].shape[0])
        print twisters
        f4 = plt.figure()
        ax = f4.add_subplot(111)
        ax.bar(ind, twisters, width, color='b')
        ax.set_ylabel('Number of Twisters out of 10,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of Twisters in Subset')
        ax.set_ylim((0, 10000))
        plt.savefig(image_path + 'subset_twisters.png', bbox_inches='tight')

        rating_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            rating_sc.append(amt[abs(amt['rating'] - amt[round_number]) < 1].shape[0])
        print rating_sc
        f5 = plt.figure()
        ax = f5.add_subplot(111)
        ax.bar(ind, rating_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 10,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of Subset Samples Where Review Rating - Sentiment Classifier is < 1')
        ax.set_ylim((0, 10000))
        plt.savefig(image_path + 'subset_rating_sent_0.png', bbox_inches='tight')

        rating_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            rating_sc.append(amt[abs(amt['rating'] - amt[round_number]) <= 1].shape[0])
        print rating_sc
        f6 = plt.figure()
        ax = f6.add_subplot(111)
        ax.bar(ind, rating_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 10,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of Subset Samples Where Review Rating - Sentiment Classifier is <= 1')
        ax.set_ylim((0, 10000))
        plt.savefig(image_path + 'subset_rating_sent_1.png', bbox_inches='tight')

        rating_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            rating_sc.append(amt[abs(amt['rating'] - amt[round_number]) <= 2].shape[0])
        print rating_sc
        f7 = plt.figure()
        ax = f7.add_subplot(111)
        ax.bar(ind, rating_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 10,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of Subset Samples Where Review Rating - Sentiment Classifier is <= 2')
        ax.set_ylim((0, 10000))
        plt.savefig(image_path + 'subset_rating_sent_2.png', bbox_inches='tight')

        print amt[abs(amt['rating'] - amt['avg_reviewer_score']) <= 1].shape[0]

        #Now for all results, not just subset with User Study annotations.

        round_number = "round1"
        output_file = self.filepath + round_number + '/' + round_number + 'matched.txt'
        review_ids = []
        ratings = []
        round_numbers = []
        # texts = []
        with open(output_file, 'r') as f:
            for line in f:
                line = line.split(';')
                review_ids.append(line[0])
                ratings.append(int(line[1]))
                round_numbers.append(int(line[2]))
        all_round_results = pd.DataFrame(zip(review_ids, ratings, round_numbers),
                                         columns=['review_id', 'rating', round_number])

        for i in range(2, 21):
            round_number = 'round' + str(i)
            output_file = self.filepath + round_number + '/' + round_number + 'matched.txt'
            review_ids = []
            round_numbers = []
            with open(output_file, 'r') as f:
                for line in f:
                    line = line.split(';')
                    review_ids.append(line[0])
                    round_numbers.append(int(line[2]))
            data = pd.DataFrame(zip(review_ids, round_numbers), columns=['review_id', round_number])
            all_round_results = pd.merge(all_round_results, data, how='left', on='review_id')
            print round_number
        '''
        all_round_results = pd.read_csv(output_file, sep=';', header=None, names=['review_id', 'rating', round_number, 'text'], usecols=['review_id', 'rating', round_number])

        for i in range(2, 21):
            round_number = 'round' + str(i)
            output_file = self.filepath + round_number + '/' + round_number + 'matched.txt'
            data = pd.read_csv(output_file, sep=';', header=None, names=['review_id', 'rating', round_number, 'text'], usecols=['review_id', round_number])
            all_round_results = pd.merge(all_round_results, data, how='left', on='review_id')
            print round_number
        '''
        all_round_results = all_round_results.dropna(subset=['round1'], how='any')

        all_round_results = all_round_results.fillna(method='pad', axis=1)
        all_round_results.to_csv('all_round_results_all.csv')

        twisters = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            twisters.append(all_round_results[abs(all_round_results['rating'] - all_round_results[round_number]) >= 2].shape[0])
        print twisters
        f8 = plt.figure()
        ax = f8.add_subplot(111)
        ax.bar(ind, twisters, width, color='b')
        ax.set_ylabel('Number of Twisters out of 60,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of Twisters in Entire Dataset')
        ax.set_ylim((0, 60000))
        plt.savefig(image_path + 'all_twisters.png', bbox_inches='tight')

        rating_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            rating_sc.append(all_round_results[abs(all_round_results['rating'] - all_round_results[round_number]) < 1].shape[0])
        print rating_sc
        f9 = plt.figure()
        ax = f9.add_subplot(111)
        ax.bar(ind, rating_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 60,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of All Samples Where Review Rating - Sentiment Classifier is < 1')
        ax.set_ylim((0, 60000))
        plt.savefig(image_path + 'all_rating_sent_0.png', bbox_inches='tight')

        rating_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            rating_sc.append(all_round_results[abs(all_round_results['rating'] - all_round_results[round_number]) <= 1].shape[0])
        print rating_sc
        f10 = plt.figure()
        ax = f10.add_subplot(111)
        ax.bar(ind, rating_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 60,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of All Samples Where Review Rating - Sentiment Classifier is <= 1')
        ax.set_ylim((0, 60000))
        plt.savefig(image_path + 'all_rating_sent_1.png', bbox_inches='tight')

        rating_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            rating_sc.append(all_round_results[abs(all_round_results['rating'] - all_round_results[round_number]) <= 2].shape[0])
        print rating_sc
        f11 = plt.figure()
        ax = f11.add_subplot(111)
        ax.bar(ind, rating_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 60,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of All Samples Where Review Rating - Sentiment Classifier is <= 2')
        ax.set_ylim((0, 60000))
        plt.savefig(image_path + 'all_rating_sent_2.png', bbox_inches='tight')

    def analyzeResultsNoSubset(self, experiment_type):
        """
        Produce plots for data that does not have a user-study annotated subset
        :param experiment_type: Level of agreement required to move a sample into training set.
        Specified so you can compare results from different runs.
        :return: None
        """
        ind = np.arange(20)
        width = 0.35
        image_path = self.filepath + 'images/' + experiment_type + '/'
        system('mkdir ' + image_path)

        round_number = "round1"
        output_file = self.filepath + round_number + '/' + round_number + 'matched.txt'
        review_ids = []
        ratings = []
        round_numbers = []
        # texts = []
        with open(output_file, 'r') as f:
            for line in f:
                line = line.split(';')
                review_ids.append(line[0])
                ratings.append(int(line[1]))
                round_numbers.append(int(line[2]))
                # texts.append(';'.join(line[3:]))
        all_round_results = pd.DataFrame(zip(review_ids, ratings, round_numbers),
                                         columns=['review_id', 'rating', round_number])
        # all_round_results = pd.read_csv(output_file, sep=';', header=None, names=['review_id', 'rating', round_number, 'text'], usecols=['review_id', 'rating', round_number])
        for i in range(2, 21):
            review_ids = []
            round_numbers = []
            round_number = 'round' + str(i)
            output_file = self.filepath + round_number + '/' + round_number + 'matched.txt'
            with open(output_file, 'r') as f:
                for line in f:
                    line = line.split(';')
                    review_ids.append(line[0])
                    ratings.append(line[1])
                    round_numbers.append(int(line[2]))
                    # texts.append(';'.join(line[3:]))
            data = pd.DataFrame(zip(review_ids, ratings, round_numbers),
                                columns=['review_id', 'rating', round_number])
            # data = pd.read_csv(output_file, sep=';', header=None, names=['review_id', 'rating', round_number, 'text'], usecols=['review_id', round_number])
            all_round_results = pd.merge(all_round_results, data, how='left', on='review_id')
            print round_number

        all_round_results = all_round_results.dropna(subset=['round1'], how='any')

        all_round_results = all_round_results.fillna(method='pad', axis=1)
        all_round_results.to_csv('all_round_results_all.csv')

        twisters = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            twisters.append(all_round_results[abs(all_round_results['rating'] - all_round_results[round_number]) >= 2].shape[0])
        print twisters
        f8 = plt.figure()
        ax = f8.add_subplot(111)
        ax.bar(ind, twisters, width, color='b')
        ax.set_ylabel('Number of Twisters out of 60,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of Twisters in Entire Dataset')
        ax.set_ylim((0, 60000))
        plt.savefig(image_path + 'all_twisters.png', bbox_inches='tight')

        rating_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            rating_sc.append(all_round_results[abs(all_round_results['rating'] - all_round_results[round_number]) < 1].shape[0])
        print rating_sc
        f9 = plt.figure()
        ax = f9.add_subplot(111)
        ax.bar(ind, rating_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 60,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of All Samples Where Review Rating - Sentiment Classifier is < 1')
        ax.set_ylim((0, 60000))
        plt.savefig(image_path + 'all_rating_sent_0.png', bbox_inches='tight')

        rating_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            rating_sc.append(all_round_results[abs(all_round_results['rating'] - all_round_results[round_number]) <= 1].shape[0])
        print rating_sc
        f10 = plt.figure()
        ax = f10.add_subplot(111)
        ax.bar(ind, rating_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 60,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of All Samples Where Review Rating - Sentiment Classifier is <= 1')
        ax.set_ylim((0, 60000))
        plt.savefig(image_path + 'all_rating_sent_1.png', bbox_inches='tight')

        rating_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            rating_sc.append(all_round_results[abs(all_round_results['rating'] - all_round_results[round_number]) <= 2].shape[0])
        print rating_sc
        f11 = plt.figure()
        ax = f11.add_subplot(111)
        ax.bar(ind, rating_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 60,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of All Samples Where Review Rating - Sentiment Classifier is <= 2')
        ax.set_ylim((0, 60000))
        plt.savefig(image_path + 'all_rating_sent_2.png', bbox_inches='tight')


    def analyze_stanford_results(self, stanford_path, experiment_type):
        """

        :param stanford_path: Location of data classified by stanford's sentiment classifier
        (http://nlp.stanford.edu/sentiment/index.html)
        :param experiment_type: Level of agreement required to move a sample into training set.
        Specified so you can compare results from different runs.
        :return: None
        """

        ind = np.arange(20)
        width = 0.35
        amt = pd.ExcelFile(self.filepath + 'userStudyResults.xlsx')
        amt = amt.parse("userStudyResults.csv")
        image_path = stanford_path + 'images/' + experiment_type + '/'
        system('mkdir ' + image_path)

        round_number = "round1"
        all_round_results = pd.DataFrame(columns={'review_id', 'rating', round_number})
        output_path = stanford_path + round_number + '/matched/'
        output_files = [f for f in os.listdir(output_path) if isfile(join(output_path, f)) and not f.startswith('.')]
        output_files.sort()

        for output in output_files:
            filename = output_path + output
            data = pd.read_csv(filename, sep=';', header=None, names=['review_id', 'rating', round_number, 'text'],
                               usecols=['review_id', 'rating', round_number])
            all_round_results = pd.concat([all_round_results, data])

        amt = pd.merge(amt, all_round_results, how='left', on='review_id')

        for i in range(2, 21):
            round_number = 'round' + str(i)
            output_path = stanford_path + round_number + '/matched/'
            output_files = [f for f in os.listdir(output_path) if isfile(join(output_path, f)) and not f.startswith('.')]
            output_files.sort()
            all_round_results = pd.DataFrame(columns={'review_id', round_number})
            for output in output_files:
                filename = output_path + output
                data = pd.read_csv(filename, sep=';', header=None, names=['review_id', 'rating', round_number, 'text'],
                                   usecols=['review_id', round_number])
                all_round_results = pd.concat([all_round_results, data])
            amt = pd.merge(amt, all_round_results, how='left', on='review_id')
            print round_number

        amt = amt.dropna(subset=['round1'], how='any')
        amt = amt.fillna(method='pad', axis=1)
        amt.to_csv(stanford_path + 'amt.csv')

        amt_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            amt_sc.append(amt[abs(amt['avg_reviewer_score'] - amt[round_number]) < 1].shape[0])
        print amt_sc
        f1 = plt.figure()
        ax = f1.add_subplot(111)
        ax.bar(ind, amt_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 10,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of Subset Samples Where User Study Results - Sentiment Classifier is < 1')
        ax.set_ylim((0, 10000))
        plt.savefig(image_path + 'subset_user_sent_0.png', bbox_inches='tight')

        amt_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            amt_sc.append(amt[abs(amt['avg_reviewer_score'] - amt[round_number]) <= 1].shape[0])
        print amt_sc
        f2 = plt.figure()
        ax = f2.add_subplot(111)
        ax.bar(ind, amt_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 10,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of Subset Samples Where User Study Results - Sentiment Classifier is <= 1')
        ax.set_ylim((0, 10000))
        plt.savefig(image_path + 'subset_user_sent_1.png', bbox_inches='tight')

        amt_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            amt_sc.append(amt[abs(amt['avg_reviewer_score'] - amt[round_number]) <= 2].shape[0])
        print amt_sc
        f3 = plt.figure()
        ax = f3.add_subplot(111)
        ax.bar(ind, amt_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 10,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of Subset Samples Where User Study Results - Sentiment Classifier is <= 2')
        ax.set_ylim((0, 10000))
        plt.savefig(image_path + 'subset_user_sent_2.png', bbox_inches='tight')

        twisters = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            twisters.append(amt[abs(amt['rating'] - amt[round_number]) >= 2].shape[0])
        print twisters
        f4 = plt.figure()
        ax = f4.add_subplot(111)
        ax.bar(ind, twisters, width, color='b')
        ax.set_ylabel('Number of Twisters out of 10,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of Twisters in Subset')
        ax.set_ylim((0, 10000))
        plt.savefig(image_path + 'subset_twisters.png', bbox_inches='tight')

        rating_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            rating_sc.append(amt[abs(amt['rating'] - amt[round_number]) < 1].shape[0])
        print rating_sc
        f5 = plt.figure()
        ax = f5.add_subplot(111)
        ax.bar(ind, rating_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 10,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of Subset Samples Where Review Rating - Sentiment Classifier is < 1')
        ax.set_ylim((0, 10000))
        plt.savefig(image_path + 'subset_rating_sent_0.png', bbox_inches='tight')

        rating_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            rating_sc.append(amt[abs(amt['rating'] - amt[round_number]) <= 1].shape[0])
        print rating_sc
        f6 = plt.figure()
        ax = f6.add_subplot(111)
        ax.bar(ind, rating_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 10,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of Subset Samples Where Review Rating - Sentiment Classifier is <= 1')
        ax.set_ylim((0, 10000))
        plt.savefig(image_path + 'subset_rating_sent_1.png', bbox_inches='tight')

        rating_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            rating_sc.append(amt[abs(amt['rating'] - amt[round_number]) <= 2].shape[0])
        print rating_sc
        f7 = plt.figure()
        ax = f7.add_subplot(111)
        ax.bar(ind, rating_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 10,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of Subset Samples Where Review Rating - Sentiment Classifier is <= 2')
        ax.set_ylim((0, 10000))
        plt.savefig(image_path + 'subset_rating_sent_2.png', bbox_inches='tight')

        print amt[abs(amt['rating'] - amt['avg_reviewer_score']) <= 1].shape[0]

        # For all results
        round_number = "round1"
        all_round_results = pd.DataFrame(columns={'review_id', 'rating', round_number})
        output_path = stanford_path + round_number + '/matched/'
        output_files = [f for f in os.listdir(output_path) if isfile(join(output_path, f)) and not f.startswith('.')]
        output_files.sort()

        for output in output_files:
            filename = output_path + output
            data = pd.read_csv(filename, sep=';', header=None, names=['review_id', 'rating', round_number, 'text'],
                               usecols=['review_id', 'rating', round_number])
            # data[round_number] = [float(x.strip()) for x in data[round_number].values]
            all_round_results = pd.concat([all_round_results, data])

        for i in range(2, 21):
            round_number = 'round' + str(i)
            output_path = stanford_path + round_number + '/matched/'
            output_files = [f for f in os.listdir(output_path) if isfile(join(output_path, f)) and not f.startswith('.')]
            output_files.sort()
            tmp = pd.DataFrame(columns={'review_id', round_number})
            for output in output_files:
                filename = output_path + output
                data = pd.read_csv(filename, sep=';', header=None, names=['review_id', 'rating', round_number, 'text'],
                                   usecols=['review_id', round_number])
                tmp = pd.concat([tmp, data])
            all_round_results = pd.merge(all_round_results, tmp, how='left', on='review_id')

            print round_number

        all_round_results = all_round_results.dropna(subset=['round1'], how='any')

        all_round_results = all_round_results.fillna(method='pad', axis=1)
        all_round_results.to_csv(stanford_path + 'all_round_results_all.csv')

        twisters = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            twisters.append(all_round_results[abs(all_round_results['rating'] - all_round_results[round_number]) >= 2].shape[0])
        print twisters
        f8 = plt.figure()
        ax = f8.add_subplot(111)
        ax.bar(ind, twisters, width, color='b')
        ax.set_ylabel('Number of Twisters out of 60,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of Twisters in Entire Dataset')
        ax.set_ylim((0, 60000))
        plt.savefig(image_path + 'all_twisters.png', bbox_inches='tight')

        rating_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            rating_sc.append(all_round_results[abs(all_round_results['rating'] - all_round_results[round_number]) < 1].shape[0])
        print rating_sc
        f9 = plt.figure()
        ax = f9.add_subplot(111)
        ax.bar(ind, rating_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 60,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of All Samples Where Review Rating - Sentiment Classifier is < 1')
        ax.set_ylim((0, 60000))
        plt.savefig(image_path + 'all_rating_sent_0.png', bbox_inches='tight')

        rating_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            rating_sc.append(all_round_results[abs(all_round_results['rating'] - all_round_results[round_number]) <= 1].shape[0])
        print rating_sc
        f10 = plt.figure()
        ax = f10.add_subplot(111)
        ax.bar(ind, rating_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 60,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of All Samples Where Review Rating - Sentiment Classifier is <= 1')
        ax.set_ylim((0, 60000))
        plt.savefig(image_path + 'all_rating_sent_1.png', bbox_inches='tight')

        rating_sc = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            rating_sc.append(all_round_results[abs(all_round_results['rating'] - all_round_results[round_number]) <= 2].shape[0])
        print rating_sc
        f11 = plt.figure()
        ax = f11.add_subplot(111)
        ax.bar(ind, rating_sc, width, color='b')
        ax.set_ylabel('Number in Agreement out of 60,000')
        ax.set_xlabel('Training Iteration')
        ax.set_title('Number of All Samples Where Review Rating - Sentiment Classifier is <= 2')
        ax.set_ylim((0, 60000))
        plt.savefig(image_path + 'all_rating_sent_2.png', bbox_inches='tight')

    def track_twisters(self):
        """
        See how training and testing set sizes change over the training rounds.
        Writes output file containing the reviews that cannot be correctly classified after 20 rounds.
        :return: None
        """

        round_number = "round1"
        output_file = self.filepath + round_number + '/' + round_number + 'matched.txt'
        review_ids = []
        ratings = []
        round_numbers = []
        with open(output_file, 'r') as f:
            for line in f:
                line = line.split(';')
                review_ids.append(line[0])
                ratings.append(int(line[1]))
                round_numbers.append(int(line[2]))
        all_round_results = pd.DataFrame(zip(review_ids, ratings, round_numbers),
                                         columns=['review_id', 'rating', round_number])

        for i in range(2, 21):
            round_number = 'round' + str(i)
            output_file = self.filepath + round_number + '/' + round_number + 'matched.txt'
            review_ids = []
            round_numbers = []
            with open(output_file, 'r') as f:
                for line in f:
                    line = line.split(';')
                    review_ids.append(line[0])
                    round_numbers.append(int(line[2]))
            data = pd.DataFrame(zip(review_ids, round_numbers), columns=['review_id', round_number])
            all_round_results = pd.merge(all_round_results, data, how='left', on='review_id')
            print round_number

        all_round_results = all_round_results.dropna(subset=['round1'], how='any')
        all_round_results = all_round_results.fillna(method='pad', axis=1)
        texts = []
        review_ids = []
        with open(output_file, 'r') as f:
            for line in f:
                line = line.split(';')
                review_ids.append(line[0])
                texts.append(';'.join(line[3:]).strip())
        reviews = pd.DataFrame(zip(review_ids, texts), columns=['review_id', 'text'])
        all_round_results = pd.merge(all_round_results, reviews, how='left', on='review_id')

        twisters = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            try:
                twisters.append(all_round_results[abs(all_round_results['rating'] - all_round_results[round_number]) >= 2][
                                    ['review_id', 'rating', round_number, 'text']])
            except Exception as e:
                print all_round_results['rating']
                print all_round_results[round_number]
        for i in range(0, len(twisters)-1):
            j = i + 1
            print "Number of twisters in round " + str(i) + ': ' + str(len(twisters[i]))
            print "Number of twisters in round " + str(j) + ': ' + str(len(twisters[j]))
            print "Overlap between rounds: " + str(len(set(twisters[i]) and set(twisters[j])))
        with open(self.filepath + 'twisters.txt', 'w') as f:
            for twister in twisters[j].values:
                for item in twister:
                    f.write(str(item) + ';')
                f.write('\n')

    @staticmethod
    def get_stanford_twisters(stanford_path):
        """
        See how training and testing set sizes change over the training rounds for the output of
        the Stanford classifier. (http://nlp.stanford.edu/sentiment/index.html)
        Writes output file containing the reviews that cannot be correctly classified after 20 rounds.
        :param stanford_path: Location of data classified by Stanford's sentiment classifier
        :return: None
        """
        round_number = "round1"
        all_round_results = pd.DataFrame(columns={'review_id', 'rating', round_number})
        reviews = pd.DataFrame(columns={'review_id', 'text'})
        output_path = stanford_path + round_number + '/matched/'
        output_files = [f for f in os.listdir(output_path) if isfile(join(output_path, f)) and not f.startswith('.')]
        output_files.sort()

        for output in output_files:
            filename = output_path + output
            data = pd.read_csv(filename, sep=';', header=None, names=['review_id', 'rating', round_number, 'text'],
                               usecols=['review_id', 'rating', round_number])
            # data[round_number] = [float(x.strip()) for x in data[round_number].values]
            all_round_results = pd.concat([all_round_results, data])
            data2 = pd.read_csv(filename, sep=';', header=None, names=['review_id', 'rating', round_number, 'text'],
                                usecols=['review_id', 'text'])
            reviews = pd.concat([reviews, data2])

        for i in range(2, 21):
            round_number = 'round' + str(i)
            output_path = stanford_path + round_number + '/matched/'
            output_files = [f for f in os.listdir(output_path) if isfile(join(output_path, f)) and not f.startswith('.')]
            output_files.sort()
            tmp = pd.DataFrame(columns={'review_id', round_number})
            for output in output_files:
                filename = output_path + output
                data = pd.read_csv(filename, sep=';', header=None, names=['review_id', 'rating', round_number, 'text'],
                                   usecols=['review_id', round_number])
                # data[round_number] = [float(x.strip()) for x in data[round_number].values]
                tmp = pd.concat([tmp, data])
            all_round_results = pd.merge(all_round_results, tmp, how='left', on='review_id')
            print round_number

        all_round_results = all_round_results.dropna(subset=['round1'], how='any')

        all_round_results = all_round_results.fillna(method='pad', axis=1)
        all_round_results = pd.merge(all_round_results, reviews, how='left', on='review_id')
        all_round_results.to_csv(stanford_path + 'all_round_results_all.csv')

        twisters = []
        for i in range(1, 21):
            round_number = 'round' + str(i)
            try:
                twisters.append(all_round_results[abs(all_round_results['rating'] - all_round_results[round_number]) >= 2]
                                [['review_id', 'rating', round_number, 'text']])
            except Exception as e:
                print e
                print all_round_results['rating']
                print all_round_results[round_number]
        for i in range(0, len(twisters)-1):
            j = i + 1
            print "Number of twisters in round " + str(i) + ': ' + str(len(twisters[i]))
            print "Number of twisters in round " + str(j) + ': ' + str(len(twisters[j]))
            print "Overlap between rounds: " + str(len(set(twisters[i]['review_id']) & set(twisters[j]['review_id'])))
        with open(stanford_path + 'twisters.txt', 'w') as f:
            for twister in twisters[j].values:
                for item in twister:
                    f.write(str(item) + ';')
                f.write('\n')

    def get_subset_twisters_no_stanford(self):
        """
        See how training and testing set sizes change over the training rounds for the annotated subset.
        Writes output file containing the subset reviews that cannot be correctly classified after 20 rounds.
        :return: None
        """
        amt = pd.ExcelFile(self.filepath + 'userStudyResults.xlsx')
        amt = amt.parse("userStudyResults.csv")
        output_file = self.filepath + 'twisters.txt'
        review_ids = []
        ratings = []
        round_20s = []
        texts = []
        with open(output_file, 'r') as f:
            for line in f:
                line = line.split(';')
                review_ids.append(line[0])
                ratings.append(int(line[1]))
                round_20s.append(float(line[2]))
                texts.append(';'.join(line[3:]).strip().strip(';'))
        nb = pd.DataFrame(zip(review_ids, ratings, round_20s, texts),
                                         columns=['review_id', 'rating', 'round20', 'text'])
        subset_ids = set(amt['review_id']) & set(nb['review_id'])
        common_core_subset_reviews = pd.merge(amt[amt['review_id'].isin(subset_ids)][['review_id', 'avg_reviewer_score']],
                                              nb[['review_id', 'rating', 'round20', 'text']], how='left', on='review_id')
        common_core_subset_reviews.to_csv(self.filepath + 'common_subset_twisters.csv', sep=';', index=False)

    def compare_twisters(self, stanford_path):
        """
        Comparing incorrectly classified data from Stanford classifier and Naive Bayes classifier
        :param stanford_path: Location of data classified by Stanford's sentiment classifier
        :return: None
        """
        amt = pd.ExcelFile(self.filepath + 'userStudyResults.xlsx')
        amt = amt.parse("userStudyResults.csv")
        nb = pd.read_csv(self.filepath + 'twisters.txt', sep=';', header=None,
                         names=['review_id', 'rating', 'round20', 'text', 'ignore'],
                         usecols=['review_id', 'rating', 'round20', 'text'])
        st = pd.read_csv(stanford_path + 'twisters.txt', sep=';', header=None,
                         names=['review_id', 'rating', 'round20', 'text', 'ignore'],
                         usecols=['review_id', 'rating', 'round20', 'text'])
        print len(set(nb['review_id']))
        print len(set(st['review_id']))
        common_core = set(nb['review_id']) & set(st['review_id'])
        common_core_subset = set(amt['review_id']) & common_core
        common_core_reviews = pd.merge(st[st['review_id'].isin(common_core)][['review_id', 'rating', 'round20']],
                                       nb[['review_id', 'round20', 'text']], how='left', on='review_id')
        common_core_subset_reviews = pd.merge(st[st['review_id'].isin(common_core_subset)][['review_id', 'rating', 'round20']],
                                              nb[['review_id', 'round20', 'text']], how='left', on='review_id')
        common_core_subset_reviews = pd.merge(common_core_subset_reviews, amt[['review_id', 'avg_reviewer_score']],
                                              how='left', on='review_id')
        only_st = st[~st['review_id'].isin(common_core)][['review_id', 'rating', 'round20', 'text']]
        only_nb = nb[~nb['review_id'].isin(common_core)][['review_id', 'rating', 'round20', 'text']]

        common_core_reviews.to_csv(self.filepath + 'common_twisters.csv', sep=';', index=False)
        common_core_subset_reviews.to_csv(self.filepath + 'common_subset_twisters.csv', sep=';', index=False)
        only_st.to_csv(self.filepath + 'stanford_twisters.csv', sep=';', index=False)
        only_nb.to_csv(self.filepath + 'nb_twisters.csv', sep=';', index=False)

        common_core = str(len(set(nb['review_id']) & set(st['review_id'])))
        print common_core

    def examine_common_subset(self):
        """
        Comparing incorrectly classified data from Stanford classifier and Naive Bayes classifier in the annotated subset.
        :return: None
        """
        subset = pd.read_csv(self.filepath + 'common_subset_twisters_renamed.csv', sep=',')
        # subset[['avg_reviewer_score', 'Stanford', 'Naive Bayes']] = subset[['avg_reviewer_score', 'Stanford', 'Naive Bayes']].astype(float)
        subset['stan_amt_dif'] = abs(subset['avg_reviewer_score'] - subset['Stanford'])
        subset['nb_amt_dif'] = abs(subset['avg_reviewer_score'] - subset['Naive Bayes'])
        subset['stan_rating_dif'] = abs(subset['rating'] - subset['Stanford'])
        subset['nb_rating_dif'] = abs(subset['rating'] - subset['Naive Bayes'])
        subset['rating_amt_dif'] = abs(subset['rating'] - subset['avg_reviewer_score'])
        subset['stan_better_nb_amt'] = subset['stan_amt_dif'] < subset['nb_amt_dif']
        subset['stan_better_nb_rating'] = subset['stan_rating_dif'] < subset['nb_rating_dif']
        subset['stan_better_rating'] = subset['stan_rating_dif'] < subset['rating_amt_dif']
        subset['nb_better_rating'] = subset['nb_rating_dif'] < subset['rating_amt_dif']
        subset.to_csv(self.filepath + 'analysis_of_agreement.csv', sep=',', index=False)


def main():
    """

    :return:
    """
    if len(sys.argv) >= 2:
        filepath = sys.argv[1]
    else:
        filepath = ''
    nbsc = NBSC(filepath)
    nbsc.round0()
    nbsc.run_experiment_iterations()
    #nbsc.track_twisters()
    #nbsc.get_subset_twisters_no_stanford()
    #nbsc.analyzeResults('exact_noMI')
    #nbsc.analyzeResultsNoSubset('exact_noMI')
    #nbsc.analyzeResultsNoSubset('one_noMI')
    #nbsc.feature_selection_trials()
    #NBSC.get_stanford_twisters('/Users/ajm/Git/sentimentanalysis/stanford/TA_Experiments/')
    #nbsc.compare_twisters('/Users/ajm/Git/sentimentanalysis/stanford/TA_Experiments/')
    #nbsc.examine_common_subset()
    #nbsc.analyze_stanford_results('/Users/ajm/Git/sentimentanalysis/stanford/TA_Experiments/', 'exact_noMI')

if __name__ == "__main__":
    main()
