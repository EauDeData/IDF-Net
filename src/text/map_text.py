import gensim
import numpy as np
from typing import *
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import common_texts
import os

from src.utils.errors import *
from src.dataloaders.base import IDFNetDataLoader
from src.text.preprocess import StringCleaner, StringCleanAndTrim

# https://open.spotify.com/track/2QLrYSnATJYuRThUZtdhH3?si=faed86630309414a

def yieldify_dataiter(dataiter: Callable, function: Callable):
    '''
    Maybe we should just use map() Python function.
    But I'm afraid I want to be able to control threads given the opportunity.
    '''
    for data in dataiter:
        yield function([data])

def _train_precondition(obj) -> None:
    if isinstance(obj.model, int): raise(ModelNotTrainedError)

class TF_IDFLoader:
    '''
    Given a dataset; loads its TF-IDF representation.
    self.fit: Builds the TF-IDF model.
    self.infer: Infers the TF-IDF representation of a new text.
    '''

    name = 'tf-idf_mapper'

    def __init__(self, dataset: IDFNetDataLoader, string_preprocess: Callable = StringCleanAndTrim, *args, **kwargs) -> None:
        self.dataset = dataset
        self.prep = string_preprocess
        self.model = 0 # IF it's an int, a not trained error will be rised

    def __getitem__(self, index: int) -> np.ndarray:
        _train_precondition(self)
        return self.model[self.corpus[0]]

    def fit(self) -> None:

        dataset = yieldify_dataiter(self.dataset.iter_text(), self.prep)
        sentences = self.prep([' '.join(x[0]) for x in dataset])
        self.dct = gensim.corpora.Dictionary(sentences)
        self.corpus = [self.dct.doc2bow(line) for line in sentences]
        self.model = gensim.models.TfidfModel(self.corpus, smartirs='ntc')

    def infer(self, index: int) -> Dict:

        return {
            "result": self[index]
        }

class BOWLoader:
    name = 'BOW_mapper'
    def __init__(self, dataset: IDFNetDataLoader, *args, **kwargs) -> None:
        pass

class LSALoader:
    name = 'LSA_mapper'
    def __init__(self, dataset: IDFNetDataLoader, string_preprocess: Callable = StringCleanAndTrim, word_to_vect: str = 'word2vect', lsa: str = 'svd') -> None:
        
        self.dataset = dataset
        self.prep = string_preprocess
        self.model = 0 # IF it's an int, a not trained error will be rised
        self.w2v_model = word_to_vect
        self.lsa = lsa

    def __getitem__(self, index: int) -> np.ndarray:
        _train_precondition(self)

        if self.lsa == 'svd':
            words = np.array([self.model[w] for w in self.corpus[index]])
            _, document, _ = np.linalg.svd(words) # Get Eigenvalues of the SVD
            return document

        else: raise InvalidModelNameError

    def fit(self):

        dataset = yieldify_dataiter(self.dataset.iter_text(), self.prep)
        sentences = [' '.join(x[0]) for x in dataset]
        self.corpus = self.prep(sentences)
        if self.w2v_model == 'word2vect':

            datapath = f'dataset/w2v_{self.dataset.name}.wordvectors'
            if os.path.exists(datapath): 
                self.model = KeyedVectors.load(datapath, mmap='r')

            else:

                model = Word2Vec(sentences=self.corpus, vector_size=100, window=5, min_count=1, workers=4)
                model.train(self.corpus, total_examples=len(self.dataset), epochs=1)
                self.model = model.wv
                self.model.save(datapath)

        else: raise InvalidModelNameError

