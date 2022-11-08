import string
import nltk
import re
from typing import *
from gensim.utils import simple_preprocess
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# https://open.spotify.com/track/31i56LZnwE6uSu3exoHjtB?si=1e5e0d5080404042

class StringCleaner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, batch: List[str], *args: Any, **kwds: Any) -> List[List]:

        '''
        Call function for string cleaner. Receives a batch of strings and cleanses them.
        Args:
            batch: list fo strings to clean
        returns:
            list of cleansed strings of format [[w1, w2, ..., wn], d2, ..., dn].
                w: words
                d: documents of batch
        '''

        return [simple_preprocess(doc) for doc in batch]

class StringCleanAndTrim:
    def __init__(self, stemm = True) -> None:
        self.stemm = stemm

    def __call__(self, batch, *args: Any, **kwds: Any) -> Any:
        '''
        Call function for string cleaner and trimmer. Receives a batch of strings and cleanses them.
        Steps: 
            0. Lower Case
            1. Lemmatization
            2. Punctation

        Args:
            batch: list fo strings to clean
        returns:
            list of cleansed strings of format [[w1, w2, ..., wn], d2, ..., dn].
                w: words
                d: documents of batch
        '''
        stopwords = nltk.corpus.stopwords.words('english')
        basic_cleaner = StringCleaner()

        shorter = PorterStemmer().stem if self.stemm else WordNetLemmatizer().lemmatize


        unstopable = [[word for word in word_tokenize(text.lower()) if not word in stopwords] for text in batch]
        cleaner = [" ".join([y for y in x if not y in string.punctuation]) for x in unstopable] # Remove punctation and lower
        shorty = [shorter(x) for x in cleaner]
        
        return basic_cleaner(shorty)

if __name__ == '__main__':
    cleaner_obj = StringCleanAndTrim()
    print(cleaner_obj(["I'm Diffie, congrats!", "Hello, why?", "Nick likes to play football, however he is not too fond of tennis."]))
