#!/usr/local/bin/python3

import sys 
import getopt
import subprocess
import itertools
import re
import math
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
    Calculates Inverse Document Frequency (IDF) for each word
    and updates the IDF denominator
    For each word, IDF = log ((Total documents) / (Ocurrences))
'''
def idfScore (wordList):
    totaldocs = len(corpus) + 1
    words= list(Counter(wordList))
    idfscore = {}
    for word in words:
        idfDenominators[word] = idfDenominators.get(word, 0) + 1
        idfscore[word] = math.log(totaldocs/idfDenominators[word], 10)
    return idfscore


'''
    Given the TF score and the IDF score, calculates de TF-IDF score,
    stores the three(3) scores in a dictionary
'''
def tfidfScore (tfscore, idfscore):
    tfidfscore = {}
    for word in tfscore.keys():
        newObj = {}
        newObj['tfscore'] = tfscore[word]
        newObj['idfscore'] = idfscore[word],
        newObj['tfidfscore'] = tfscore[word] * idfscore[word]
        tfidfscore[word] = newObj
    return tfidfscore


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
    fullDocument = subprocess.check_output(['curl', link])
    # Cleans document (returns list with body's words)
    wordList = cleanDocument(fullDocument)
    # Calculate TF-IDF score (returns dictionary)
    tfscore = tfScore(wordList)
    idfscore = idfScore(wordList)
    tfidfscore = tfidfScore(tfscore, idfscore)

    return title, link, tfidfscore

def incrementIdfDenominators (words):
    return

def addTFIDF (title, link, tfScore):
    return 

def addDoc(title,link, tfidfscore):
    doc = {
        'title': title,
        'link': link, 
        'tfidfscore' : tfidfscore
    }
    corpus.append(doc)
    print('Added document to corpus:', title)

def parseFeed (url):
    rss = subprocess.check_output(['curl', url])
    rssTree = etree.fromstring(rss)
    items= rssTree.xpath('//item')
    for item in items:
        title, link, tfidfscore = parseItem (item)
        #TODO: updateTFIDF(tfScore)
        addDoc(title,link, tfidfscore)


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

def searchDocument (keyword):
    docList = []
    for doc in corpus: 
        if keyword in doc['tfidfscore'] :
            obj = {
                'title'     : doc['title'],
                'link'      : doc['link'],
                'tfidfscore': doc['tfidfscore'][keyword]
            }
            docList.append(obj)
    return docList

def prettyPrint (results):
    i = 0
    for r in results:
        i = i + 1
        print(str(i) + '. ========================')
        print('Title:', r['title'])
        print('Link:', r['link'])
        print('TFIDF:', str(r['tfidfscore']['tfidfscore']))
        print()


# Setup

corpus = []
idfDenominators = {}


# Initializes stemmer
porter = PorterStemmer()

# Gather command line options
opts, args = getopt.getopt(sys.argv[1:], 'ls:r')
opts = dict(opts)

if '-l' in opts:    
    corpus, idfDenominators = loadPickles()
    parseFeed(args[0])
    dumpPickles()

if '-r' in opts: # Resets pickle files
    idfPickle = open(f'./idfDenominators.pkl', 'wb+')
    corpusPickle = open(f'./corpus.pkl', 'wb+')

    pickle.dump({}, idfPickle)
    pickle.dump([], corpusPickle)

    idfPickle.close()
    corpusPickle.close()

if '-s' in opts: 
    corpus, idfDenominators = loadPickles()
    results = searchDocument(opts['-s'])
    results.sort(reverse=True, key=(lambda x: x['tfidfscore']['tfidfscore']))
    prettyPrint(results)
