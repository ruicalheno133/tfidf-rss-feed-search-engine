#!/usr/local/bin/python3

'''
    RSS FEED TF-IDF BASED SEARCH ENGINE 

    Script for loading a RSS FEED (BBC News) into a corpus, 
    and querying it using TF-IDF method to rank the query results.
'''

import sys 
import getopt
import subprocess
import itertools
import re
import math
import pickle
import feedparser
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk.stem import PorterStemmer
from lxml import etree
from bs4 import BeautifulSoup
from collections import Counter

'''
    List of English StopWords
'''
stopWords = set(stopwords.words('english')) 


'''
    Removes blank spaces from a list
'''
def removeBlanks (l):
    while '' in l:
        l.remove('')
    return l

'''
    Word stemming using Porter Stemmer
'''
def stemWords (words):
    stem = []
    for word in words:
        words.append(porter.stem(word))
    print(words)
    return stem

'''
    Removes stop words from a list of words
'''
def removeStopWords (wordList):
    return list(filter(lambda x: not x in stopWords, wordList))

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
        p = str(p).lower() # lowers text
        p = re.sub(r'<p.*?>|</p>', '', p) # remove tags
        p = re.split(r'\W+', p) # remove spaces
        words.append(p)
    
    words = list(itertools.chain.from_iterable(words))
    words = removeBlanks(words) # remove espaços em branco da lista
    words = removeStopWords(words)
    # words = stemWords(words) # faz stemming
    return words

'''
    Calculates Term Frequency (TF) for each word
    For each word, TF = (Number Occurences) / (Total words)
'''
def tfScore (wordList):
    total = len(wordList)
    occur = Counter(wordList) 
    print(occur)
    for key, value in occur.items():
        occur[key] = value / total
    return occur

'''
    Given a word calculates its Inverse Document Frequency (IDF)
    IDF = log ((Total documents) / (Ocurrences))
'''
def idfScore (keyword):
        totalDocs = getInfo(corpus)[1]
        wordOccur = len(corpus[keyword])
        return math.log(totalDocs/wordOccur,10) 


'''
    Given the TF score and the IDF score, calculates de TF-IDF score
'''
def tfidfScore (tfscore, idfscore):
    return tfscore * idfscore


'''
    Parses an item tag from a xml RSS feed
    Gathers link, cleans content, gathers TF Score
'''
def parseItem (link):
    # Gathers html page
    fullDocument = subprocess.check_output(['curl','-L', link])
    # Cleans document (returns list with body's words)
    wordList = cleanDocument(fullDocument)
    
    # Calculate TF Score (returns dictionary)
    tfscore = tfScore(wordList)

    return tfscore

'''
    Adds a document to the corpus
'''
def addDoc(title,link, tfscore):
    for word, tf in tfscore.items():
        doc = {
            'title': title,
            'link': link, 
            'tfscore' : tf
        }
        if word in corpus:
            corpus[word].append(doc)
        else:
            corpus[word] = [doc]
    print('Added document to corpus:', title)

'''
    Parses a RSS feed, adding each document to the corpus 
    For each document stores its title, link and tf score
'''
def parseFeed (url):
    feed = feedparser.parse(url)
    for doc in feed.entries:
        print(doc.link)
        tfscore = parseItem (doc.link)
        addDoc(doc.title, doc.link, tfscore)


'''
    Loads pickles
'''
def loadPickles():
    corpusPickle = open(f'./pickles/corpus.pkl', 'rb+')
    # load Corpus information
    try:
        corpus = pickle.load(corpusPickle)
    except EOFError:
        corpus = []

    corpusPickle.close()

    return corpus

'''
    Dumps pickles
'''
def dumpPickles ():
    corpusPickle = open(f'./pickles/corpus.pkl', 'wb+')

    pickle.dump(corpus, corpusPickle)

    corpusPickle.close()

'''
    Given a keyword, searches for it in the corpus
    Returns the list of documents that obbey this condition
'''
def searchDocument (keyword):
    keyword = process.extractOne(keyword, corpus.keys())[0]
    print('Showing results for: ' + keyword, end='\n\n')
    docList = []
    if keyword in corpus:
        idfscore = idfScore(keyword)
        for doc in corpus[keyword]:
            tfidf = tfidfScore(doc['tfscore'], idfscore)
            if tfidf > 0.01:
                obj = {
                    'title'     : doc['title'],
                    'link'      : doc['link'],
                    'tfidf'     : tfidf
                }
                docList.append(obj)
    return docList

'''
    Pretty prints the results given
'''
def prettyPrint (results):
    i = 0
    for r in results:
        i = i + 1
        print(str(i) + '. ========================')
        print('Title:', r['title'])
        print('Link:', r['link'])
        print('TFIDF:', str(r['tfidf']))
        print()

'''
    Gather info from corpus 
    > Number of documents
    > Number of words 
'''
def getInfo (corpus):
    nwords = 0
    ndocs = 0
    docList = {}
    for word, docs in corpus.items():
        nwords += 1
        for doc in docs:
            docList[doc['title']] = 1
    ndocs = len(docList.keys())
    return ndocs, nwords
    

'''
    Setup
'''

''' Initializes corpus '''
corpus = []

''' Initializes Porter stemmer'''
porter = PorterStemmer()

''' Gather command line information'''
opts, args = getopt.getopt(sys.argv[1:], 'ls:rcip:')
opts = dict(opts)

'''
    Loads a rss feed into the corpus
'''
if '-l' in opts:
    print('Loading corpus...') 
    corpus = loadPickles()
    print('Corpus loaded.')
    parseFeed(args[0])
    print('Parsing feed...')
    dumpPickles()
    print('Done.')

'''
    Resets/Initializes corpus 
'''
if '-r' in opts: # Resets pickle files
    print("Reseting pickle files")
    corpusPickle = open(f'./pickles/corpus.pkl', 'wb+')
    pickle.dump({}, corpusPickle)
    corpusPickle.close()
    print("Done.")

'''
    Runs a "One word" search query 
    Results are based on the current tfidf value
'''
if '-s' in opts: 
    corpus = loadPickles()
    results = searchDocument(opts['-s'].lower())
    results.sort(reverse=True, key=(lambda x: x['tfidf']))
    prettyPrint(results)

'''
    Displays information about the current Corpus 
    Information contains:
        - Number of documents,
        - Number of words
'''
if '-i' in opts:
    corpus = loadPickles() 
    info = getInfo(corpus)
    print(info)

if '-p' in opts:
    corpus = loadPickles() 
    print(corpus[opts['-p']])
