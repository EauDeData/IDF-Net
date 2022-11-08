import gensim
import numpy as np
from typing import *

from src.utils.errors import *
from src.dataloaders.base import IDFNetDataLoader
from src.text.preprocess import StringCleaner

# https://open.spotify.com/track/2QLrYSnATJYuRThUZtdhH3?si=faed86630309414a

def yieldify_dataiter(dataiter: Callable, function: Callable):
    '''
    Maybe we should just use map() Python function.
    But I'm afraid I want to be able to control threads given the opportunity.
    '''
    for data in dataiter:
        yield function(data)

class TF_IDFLoader:
    '''
    Given a dataset; loads its TF-IDF representation.
    self.fit: Builds the TF-IDF model.
    self.infer: Infers the TF-IDF representation of a new text.
    '''

    name = 'tf-idf_mapper'

    def _train_precondition(self) -> None:
        if isinstance(self.model, int): raise(ModelNotTrainedError)

    def __init__(self, dataset: IDFNetDataLoader, string_preprocess: Callable = StringCleaner, *args, **kwargs) -> None:
        self.dataset = dataset
        self.prep = string_preprocess
        self.model = 0 # IF it's an int, a not trained error will be rised

    def __getitem__(self, index: int) -> np.ndarray:
        self._train_precondition()
        return self.model[self.corpus[index]]

    def fit(self) -> None:

        dataset = yieldify_dataiter(self.dataset.iter_text(), self.prep) # TODO: NCols (number of terms) if HUGE. Reduce to NN feasible manner.
        dct = gensim.corpora.Dictionary(dataset)
        self.corpus = [dct.doc2bow(line) for line in dataset]
        self.model = gensim.models.TfidfModel(self.corpus)

    def infer(self, index: int) -> Dict:

        return {
            "result": self[index]
        }

class BOWLoader:
    name = 'BOW_mapper'
    def __init__(self, dataset: IDFNetDataLoader, *args, **kwargs) -> None:
        pass

