import gensim
import gensim.downloader as api
import gensim.corpora as corpora
import numpy as np
from typing import *
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import common_texts
import os
import clip
import torch
from transformers import BertTokenizer, BertModel

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
        instance = self.model[self.corpus[index]]
        gensim_vector = gensim.matutils.sparse2full(instance, len(self.dct))
        return gensim_vector

    def fit(self) -> None:

        dataset = yieldify_dataiter(self.dataset.iter_text(), self.prep)
        sentences = self.prep([' '.join(x[0]) for x in dataset])
        self.dct = gensim.corpora.Dictionary(documents=sentences)
        self.corpus = [self.dct.doc2bow(line) for line in sentences]
        self.model = gensim.models.TfidfModel(self.corpus, smartirs='ntc')

    def predict(self, sentence):
        new_text_corpus =  self.dct.doc2bow(sentence.split())
        return gensim.matutils.sparse2full(self.model[new_text_corpus], len(self.dct))

    def infer(self, index: int) -> Dict:

        return {
            "result": self[index]
        }

class LDALoader:
    # https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation
    name = 'LDA_mapper'
    def __init__(self, dataset: IDFNetDataLoader, string_preprocess: Callable = StringCleanAndTrim, num_topics = 28):

        self.dataset = dataset
        self.prep = string_preprocess
        self.ntopics = num_topics
        self.model = 0

    def __getitem__(self, index):
        _train_precondition(self)
        instance = self.model[self.corpus[index]]
        return gensim.matutils.sparse2full(instance, self.ntopics)
    
    def predict(self, sentence):
        new_text_corpus =  self.dct.doc2bow(sentence.split())
        return gensim.matutils.sparse2full(self.model[new_text_corpus],  self.ntopics)

    def fit(self):
        dataset = yieldify_dataiter(self.dataset.iter_text(), self.prep)
        sentences = self.prep([' '.join(x[0]) for x in dataset])
        self.dct = gensim.corpora.Dictionary(documents=sentences)
        self.corpus = [self.dct.doc2bow(line) for line in sentences]
        self.model = gensim.models.LdaMulticore(corpus=self.corpus,
                                       id2word=self.dct,
                                       num_topics=self.ntopics)
    def infer(self, index):
        return {
            "result": self[index]
        }

class BOWLoader:
    name = 'BOW_mapper'
    def __init__(self, dataset: IDFNetDataLoader, *args, **kwargs) -> None:
        pass

class LSALoader:
    name = 'LSA_mapper'
    def __init__(self, dataset: IDFNetDataLoader, string_preprocess: Callable = StringCleanAndTrim, ntopics = 200) -> None:
        
        self.dataset = dataset
        self.prep = string_preprocess
        self.model = 0 # IF it's an int, a not trained error will be rised
        self.ntopics = ntopics

    def __getitem__(self, index: int) -> np.ndarray:
        _train_precondition(self)
        instance = self.model[self.corpus[index]]
        gensim_vector = gensim.matutils.sparse2full(instance, self.ntopics)
        return gensim_vector

    def predict(self, sentence):
        new_text_corpus =  self.dct.doc2bow(sentence.split())
        return gensim.matutils.sparse2full(self.model[new_text_corpus], self.ntopics)

    def fit(self):
        
        dataset = yieldify_dataiter(self.dataset.iter_text(), self.prep)
        sentences = self.prep([' '.join(x[0]) for x in dataset])
        self.dct = gensim.corpora.Dictionary(documents=sentences)

        self.corpus = [self.dct.doc2bow(line) for line in sentences]
        self.model = gensim.models.lsimodel.LsiModel(
            corpus=self.corpus, id2word=self.dct, num_topics=self.ntopics,
            )
        
class ParallelWrapper(torch.nn.Module):
    def __init__(self, fn) -> None:
        super(ParallelWrapper, self).__init__()
        self.fn = fn
    
    def forward(self, batch):
        return self.fn(batch)
    
class CLIPLoader:
    name = 'CLIP_mapper'
    def __init__(self, device = 'cuda', *args, **kwargs) -> None:
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit = False)
        self.model.eval()
        self.predict = ParallelWrapper(self._predict)
        self.encode_images = ParallelWrapper(self._encode_images)

    def _predict(self, text):
        tokens = clip.tokenize(text).to(self.device)
        return self.model.encode_text(tokens)

    def _encode_images(self, batch):
        return self.model.encode_image(batch)


class BertTextEncoder:
    def __init__(self, pretrained = 'bert-base-multilingual-cased') -> None:
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.model = BertModel.from_pretrained(pretrained)

    def predict(self, batch):
        encoded_input = self.tokenizer(batch, return_tensors='pt')
        return self.model(**encoded_input).pooler_output # (BS, 768) 
    
    def to(self, device):
        return self.model.to(device)