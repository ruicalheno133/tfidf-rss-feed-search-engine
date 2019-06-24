# TFIDF Based Search Engine for RSS feeds 

A python script that updates its corpus with the content of a RSS feed and queries its content
using the TFIDF method.

## Usage 

Load the corpus with new documents.

```
./rss_search.py -l <source>
```

Query the corpus (single word query).

```
./rss_search.py -s <word>
```

Get corpus details (Number of documents, Number of words)

```
./rss_search.py -c 
```

