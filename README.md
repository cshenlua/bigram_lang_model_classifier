# bigram_lang_model_classifier
A Python program that implements a language classification system based on bigram probabilities. The program trains a model using text data from different authors, considering sequences of two adjacent words (bigrams). Additionally, it incorporates LaPlace smoothing to handle unseen bigrams. After training, the model is capable of prediciting the authorship of sentences or text based on learned patterns. It can either be tested on a specific file or evaluated on a development set to measure the accuracy of author attribution.


# Run : 

## Without test file
```
python3 classifier.py authorList
```

## With test file
```
python3 classifier.py authorlist -test testfile
```

# Contents of ```authorlist``` : 
```
austen.txt
dickens.txt
tolstoy.txt
wilde.txt
```
## **NOTE :** The ```.txt``` files contain text that are ASCII "transliteration" s of UTF-8 encodings. 

