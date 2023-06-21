import string
import nltk
import re
from typing import *
from gensim.utils import simple_preprocess
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from src.utils.errors import *
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
    def __init__(self, stemm = True, lang = 'spanish') -> None:
        self.stemm = stemm
        self.lang = lang
        self.stopwords = nltk.corpus.stopwords.words(lang) # TODO, wtf hardcoded this, for real???


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

        shorter = SnowballStemmer(self.lang).stem if self.stemm else WordNetLemmatizer(self.lang).lemmatize
        lemma = [shorter(re.sub('[^A-Za-z0-9]+', '', x)) for x in batch.lower().split() if not x in self.stopwords]
        return lemma

