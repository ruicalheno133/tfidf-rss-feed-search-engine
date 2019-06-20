#!/usr/local/bin/python3

import sys 
import getopt
import subprocess
import itertools
import re
import pickle
from nltk.stem import PorterStemmer
from lxml import etree
from bs4 import BeautifulSoup
from collections import Counter

'''
    Removes blank spaces from a list
'''
def removeBlanks (l):
    while '' in l:
        l.remove('')
    return l

'''
    Word stemming based on Porter Stemmer
'''
def stemWords (words):
    stem = []
    for word in words:
        words.append(porter.stem(word))
    print(words)
    return stem

'''
    Cleans an html document
    Filter by paragraph, removes tags,
    splits by non alpha-numeric & stems the words

'''
def cleanDocument(doc):
    words = []
    soup = BeautifulSoup(doc, 'html.parser')
    # get paragraphs
    paragraphs = soup.find_all('p') 
    # split words 
    for p in paragraphs:
        p = str(p)
        p = re.sub(r'<p.*?>|</p>', '', p) # remove tags
        p = re.split('\W+', p) # remove spaces
        words.append(p)
    words = list(itertools.chain.from_iterable(words))
    words = removeBlanks(words) # remove espaços em branco da lista
    # words = stemWords(words) # faz stemming
    return words

'''
    Calculates Term Frequency (TF) for each word
    For each word, TF = (Number Occurences) / (Total words)
'''
def tfScore (wordList):
    total = len(wordList)
    occur = Counter(wordList) 
    for key, value in occur.items():
        occur[key] = value / total
    return occur

'''
    Parses an item tag from a xml RSS feed
    Gathers link, cleans content, gathers TF Score
'''
def parseItem (item):
    # TODO: clean titles
    # TODO: insert in database
    # TODO: parse item and calculate TF-IDF
    title = item.xpath('./title')[0].text
    link = item.xpath('./link')[0].text

    # Gathers html page
    fullDocument = subprocess.check_output(['curl', 'https://www.bbc.com/news/health-48512923'])
    # Cleans document (returns list with body's words)
    wordList = cleanDocument(fullDocument)
    # Calculate TF-IDF value (returns dictionary)
    tfscore = tfScore(wordList)

    return title, link, tfscore

def incrementIdfDenominators (words):
    for word in words:
        idfDenominators[word] = idfDenominators.get(word, 0) + 1



def parseFeed (url):
    rss = subprocess.check_output(['curl', url])
    rssTree = etree.fromstring(rss)
    items= rssTree.xpath('//item')
    for item in items:
        title, link, tfScore = parseItem (item)
        incrementIdfDenominators(tfScore.keys())
        # TODO: insert into database

# Load pickles 
def loadPickles():
    idfPickle = open(f'./idfDenominators.pkl', 'rb+')
    corpusPickle = open(f'./corpus.pkl', 'rb+')
    # load IDF denominators
    try:
        idfDenominators = pickle.load(idfPickle)
    except EOFError:
        idfDenominators = {}

    # load Corpus information
    try:
        corpus = pickle.load(corpusPickle)
    except EOFError:
        corpus = []

    idfPickle.close()
    corpusPickle.close()

    return corpus, idfDenominators

# Dump pickles
def dumpPickles ():
    idfPickle = open(f'./idfDenominators.pkl', 'wb+')
    corpusPickle = open(f'./corpus.pkl', 'wb+')

    pickle.dump(idfDenominators, idfPickle)
    pickle.dump(corpus, corpusPickle)

    idfPickle.close()
    corpusPickle.close()

# Setup

# Load pickle files


# Initializes stemmer
porter = PorterStemmer()

# Gather command line options
opts, args = getopt.getopt(sys.argv[1:], 'lc')
opts = dict(opts)

if '-l' in opts:    
    corpus, idfDenominators = loadPickles()
    parseFeed(args[0])
    dumpPickles()

if '-c' in opts: # Creates pickle files
    idfPickle = open(f'./idfDenominators.pkl', 'wb+')
    corpusPickle = open(f'./corpus.pkl', 'wb+')
    




