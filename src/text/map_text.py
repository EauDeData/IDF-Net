from typing import Callable
import gensim
import gensim.downloader as api
import gensim.corpora as corpora
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

class BaseMapper:
    model_instance = None
    popfitted = None
    def __init__(self, dataset: IDFNetDataLoader, string_preprocess: Callable = StringCleanAndTrim(), ntopics = 224, *args, **kwargs) -> None:
        self.dataset = dataset
        self.prep = string_preprocess
        self.model = 0 # IF it's an int, a not trained error will be rised
        
        self.ntopics = ntopics

    def __getitem__(self, index: int) -> np.ndarray:
        _train_precondition(self)
        instance = self.model[self.corpus[index]]
        gensim_vector = gensim.matutils.sparse2full(instance, self.vector_size)
        return gensim_vector
    def pop_fit(self, dict):
        return {x: dict[x] for x in dict if x in self.popfitted}
    
    def fit(self) -> None:
        
        sentences = [self.prep(x) for x in self.dataset.iter_text()]
        print("Dataset processed...", sentences[-1])
        self.dct = gensim.corpora.Dictionary(documents=sentences)
        print('Creating Corpus...')
        self.corpus = [self.dct.doc2bow(line) for line in sentences]
        print('Fitting...', self.corpus[-1])
        self.model = self.model_instance(**self.pop_fit({
                                    "corpus":self.corpus,
                                    "id2word":self.dct,
                                    "num_topics":self.ntopics
                                    }))
        self.vector_size = len(self.dct) if self.name == 'tf-idf_mapper' else self.ntopics

    def predict(self, sentence):
        new_text_corpus =  self.dct.doc2bow(sentence.split())
        return gensim.matutils.sparse2full(self.model[new_text_corpus], self.vector_size)

    def infer(self, index: int) -> Dict:

        return {
            "result": self[index]
        }

class TF_IDFLoader(BaseMapper):
    '''
    Given a dataset; loads its TF-IDF representation.
    self.fit: Builds the TF-IDF model.
    self.infer: Infers the TF-IDF representation of a new text.
    '''

    name = 'tf-idf_mapper'
    model_instance = gensim.models.TfidfModel
    popfitted = ['corpus'] # Accepted keywords in Fit function
     
class LDALoader(BaseMapper):
    # https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation
    name = 'LDA_mapper'
    model_instance = gensim.models.LdaMulticore
    popfitted = ['corpus', 'id2word', 'num_topics'] # Accepted keywords in Fit function

class EnsembleLDALoader:
    name = 'EnsembleLDA_mapper'
    model_instance = gensim.models.EnsembleLda
    popfitted = ['corpus', 'id2word', 'num_topics'] # Accepted keywords in Fit function

class BOWLoader:
    name = 'BOW_mapper'
    def __init__(self, dataset: IDFNetDataLoader, *args, **kwargs) -> None:
        pass

class LSALoader:
    name = 'LSA_mapper'
    model_instance = gensim.models.LsiModel
    popfitted = ['corpus', 'id2word', 'num_topics'] # Accepted keywords in Fit function

    
        
        
    