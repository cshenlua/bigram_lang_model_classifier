'''
Results when run without -test flag
training... (this may take a while)
austen  35 / 100 correct
dickens 40 / 100 correct
tolstoy 38 / 100 correct
wilde   30 / 100 correct
'''

import sys
import re
import argparse
import nltk
import math

def word_tokenize(line):
    # Tokenize line
    words = nltk.tokenize.word_tokenize(line.lower()) 
    # Remove punctuation
    for word in words: 
        if word == "." or word == "," or word == "!" or word == "?":
            del word
    return words

def sentence_tokenize(f):
    with open(f) as fp:
        text = fp.read()
        # remove newline token and punctuation from tokenized_sentence
        sents = [x.replace('\n',' ') for x in nltk.tokenize.sent_tokenize(text)]
        sents = [x.replace('.', ' ') for x in sents]
        sents = [x.replace(';', ' ') for x in sents]
        sents = [x.replace(',', ' ') for x in sents]
        sents = [x.replace('?', ' ') for x in sents]
        sents = [x.replace('!', ' ') for x in sents]
        sents = [x.replace('-', ' ') for x in sents]
        sents = [x.replace('"', ' ') for x in sents]
        sents = [x.replace(':', ' ') for x in sents]
        sents = [x.replace('_', ' ') for x in sents]
        sents = [x.replace('^', ' ') for x in sents]
    return sents   

def train(authors):
    vocabulary = open("words.txt", "r").readlines()
    # Read in the names of files to train on
    filenames = open(authors + ".txt", "r").readlines() 
    # Create a dictionary for the corpus
    corpus = {} 
    # Read and sentence tokenize each file and add it to the corpus dictionary
    for file in filenames: 
        file = re.sub(r"\n", "", file)
        name = re.sub(r".txt", "", file)
        corpus[name] = sentence_tokenize(file)
    # Create dictionary of bigrams
    bigrams = {} 
    # Create dictionary of bigram probabilities
    bigram_probs = {} 
    # Create dictionary of unigrams
    unigrams = {} 
    print("training... (this may take a while)")
    # For each author:
    for author in corpus: 
        # Add a dictionary to the bigrams with key for the author
        bigrams[author] = {} 
        # Add a dictionary to the bigram probabilities with key for the author
        bigram_probs[author] = {} 
        # Add a dictionary to the unigrams with key for the author
        unigrams[author] = {} 
        # Calculate how many sentences will be used for training
        num_sents = int(len(corpus[author])*0.8) 
        # For each sentence in each author's work:
        for s in range(0, num_sents - 1): 
            # Word tokenize the sentence
            tokenized_words = word_tokenize(corpus[author][s]) 
            # For each word:
            for i in range(0, len(tokenized_words) - 1): 
                # If the bigram containing the word and the next word exists in the dictionary, increment count. 
                # Else, add it to the dictionary.
                if (tokenized_words[i], tokenized_words[i + 1]) in bigrams[author]: 
                    bigrams[author][(tokenized_words[i], tokenized_words[i + 1])] += 1
                else:
                    # Add each new bigram to the probabilities dictionary
                    bigram_probs[author][(tokenized_words[i], tokenized_words[i + 1])] = 1 
                    bigrams[author][(tokenized_words[i], tokenized_words[i + 1])] = 1 
                # If the unigram containing the word exists in the dictionary, increment count. 
                # Else, add it to the dictionary.
                if tokenized_words[i] in unigrams[author]: 
                    unigrams[author][tokenized_words[i]] += 1
                else:
                    unigrams[author][tokenized_words[i]] = 1
    for author in corpus:
        V = len(unigrams[author])
        for bigram in bigram_probs[author]:
            bigram_probs[author][bigram] = math.log((bigrams[author][bigram] + 1) / (unigrams[author][bigram[0]] + V))
           
    scores = development(name, corpus, bigrams, bigram_probs, unigrams)
    for author in corpus:
        print(author + "\t" + str(scores[author]) + " / 100 correct")

def good_turing(bigrams, unigrams):
    bigrams_per_freq = {}
    for author in bigrams:
        bigrams_per_freq[author] = dict()
        # Track the frequencies of bigram frequencies
        for bigram in bigrams[author]: 
            if bigrams[author][bigram] in bigrams_per_freq[author]:
                bigrams_per_freq[author][bigram] += 1
            else:
                bigrams_per_freq[author][bigram] = 1
    print(bigrams_per_freq["austen"])

def development(name, corpus, bigrams, bigram_probs, unigrams):
    scores = {}
    for author in corpus:
        scores[author] = 0
    # For each author
    for author in corpus: 
        for sent in range(int(len(corpus[author])*0.8) + 1, len(corpus[author]) - 1):
            if sent < len(corpus[author]) - 1:
                # Tokenize the sentence
                words = word_tokenize(corpus[author][sent]) 
                probs = {}
                # For each author:
                for authorname in bigram_probs: 
                    probs[authorname] = 1
                    for w in range(0, len(words) - 1):
                        if (words[w], words[w + 1]) in bigram_probs[authorname]:
                            # Multiply the current probability of the phrase by that of the next bigram
                            probs[authorname] *= bigram_probs[authorname][(words[w], words[w + 1])] 
                max = author
                # Iterate through authors and determine which was most probable
                for a in probs: 
                    if probs[a] > probs[max]:
                        max = a
                # Check if prediction was correct
                if max == author: 
                    scores[author] += 1
    for author in corpus: 
        # Convert score to a number out of 100
        scores[author] = int((scores[author] / (len(corpus[author]) * 0.2)) * 100) 
    return scores # Return scores 
    
def train_file(authors, filename):
    vocabulary = open("words.txt", "r").readlines()
    # Read in the names of files to train on
    filenames = open(authors + ".txt", "r").readlines() 
    # Create a dictionary for the corpus
    corpus = {} 
    sentences = open(filename + ".txt", "r").readlines()
    # Read and sentence tokenize each file and add it to the corpus dictionary
    for file in filenames: 
        file = re.sub(r"\n", "", file)
        name = re.sub(r".txt", "", file)
        corpus[name] = sentence_tokenize(file)
    # Create dictionary of bigrams
    bigrams = {} 
    # Create dictionary of bigram probabilities
    bigram_probs = {} 
    # Create dictionary of unigrams
    unigrams = {} 
    print("training... (this may take a while)")
    for author in corpus:
        # Add a dictionary to the bigrams with key for the author
        bigrams[author] = {} 
        # Add a dictionary to the bigram probabilities with key for the author
        bigram_probs[author] = {} 
        # Add a dictionary to the unigrams with key for the author
        unigrams[author] = {} 
        # Calculate how many sentences will be used for training
        num_sents = len(corpus[author]) 
        for s in range(0, num_sents - 1):
            # Word tokenize each sentence
            tokenized_words = word_tokenize(corpus[author][s]) 
            for i in range(0, len(tokenized_words) - 1):
                # If the bigram containing the word and the next word exists in the dictionary, increment count. 
                # Else, add it to the dictionary.
                if (tokenized_words[i], tokenized_words[i + 1]) in bigrams[author]: 
                    bigrams[author][(tokenized_words[i], tokenized_words[i + 1])] += 1
                else:
                    # Add each new bigram to the probabilities dictionary
                    bigram_probs[author][(tokenized_words[i], tokenized_words[i + 1])] = 1 
                    bigrams[author][(tokenized_words[i], tokenized_words[i + 1])] = 1
                # If the unigram is in the dictionary, increment it. Else, add it.
                if tokenized_words[i] in unigrams[author]: 
                    unigrams[author][tokenized_words[i]] += 1
                else:
                    unigrams[author][tokenized_words[i]] = 1
    for author in corpus:
        # Length of vocabulary
        V = len(unigrams[author]) 
        for bigram in bigram_probs[author]:
            bigram_probs[author][bigram] = math.log((bigrams[author][bigram] + 1) / (unigrams[author][bigram[0]] + V)) #Calculate bigram probabilities
    for sentence in sentences:
        # Word tokenize each sentence
        words = word_tokenize(sentence) 
        probs = {}
        for author in bigram_probs:
            probs[author] = 1
            # Calculate probability of the sentence for each author
            for w in range(0, len(words) - 1): 
                if (words[w], words[w + 1]) in bigram_probs[author]:
                    probs[author] *= bigram_probs[author][(words[w], words[w + 1])]
        max = name
        # Find the most probable author
        for author in probs: 
            if probs[author] > probs[max]:
                max = author
        # Print result
        print(max) 

def main():
    # Create and argument parser
    parser = argparse.ArgumentParser() 
    parser.add_argument("authors")
    parser.add_argument("-test", "--test", type = str)
    args = parser.parse_args()
    # If "-test" is included, test on provided file
    # Otherwise, create a dev set to test
    if args.test: 
        train_file(args.authors, args.test)
    else: 
        train(args.authors)

if __name__ == '__main__':
    main()
