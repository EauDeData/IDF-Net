from typing import Callable
import gensim
import gensim.downloader as api
import gensim.corpora as corpora
import numpy as np
from typing import *
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import common_texts
import os
from transformers import BertTokenizer, BertModel
from torchtext.data import get_tokenizer
import math

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
        sentence_cleaned = self.prep(sentence)
        new_text_corpus =  self.dct.doc2bow(sentence_cleaned)
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

class LSALoader(BaseMapper):
    name = 'LSA_mapper'
    model_instance = gensim.models.LsiModel
    popfitted = ['corpus', 'id2word', 'num_topics'] # Accepted keywords in Fit function


class BertTextEncoder:
    def __init__(self, pretrained = 'bert-base-multilingual-cased') -> None:
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.model = BertModel.from_pretrained(pretrained)

    def predict(self, batch):
        encoded_input = self.tokenizer(batch, return_tensors='pt', padding=True).to(self.model.device)
        return self.model(**encoded_input).pooler_output # (BS, 768) 
    
    def to(self, device):
        self.model = self.model.to(device)
        return self

class TextTokenizer:
    bos = '<BOS>'
    eos = '<EOS>'
    unk = '<UNK>'
    pad = '<PAD>'

    ret = '<RET>' # The retrieval token is god
    rank = '<RNK>'

    per = '<PER>'
    org = '<ORG>'
    misc = '<MISC>'
    loc = '<LOC>'

    def __init__(self, cleaner) -> None:
        self.cleaner = cleaner
        self.tokens = None
        self.max_seq_size = 3000

    def predict(self, text: str):

        tokens = self.cleaner(text)
        tokens = [self.ret] + [self.rank] + [self.bos] + tokens + [self.eos]

        vector = np.zeros(len(tokens))
        for n, token in enumerate(tokens): vector[n] = self.tokens[token] if token in self.tokens else self.tokens[self.unk]

        return vector[:self.max_seq_size]

    def fit(self, dataset):
        freqs = {}
        self.min_leng = 1
        for sntc in dataset.iter_text():
            tokens = self.cleaner(sntc)
            if len(tokens) > self.min_leng: self.min_leng = len(tokens)
            for token in tokens: 
                if token not in freqs: freqs[token] = 0
                freqs[token] += 1
        
        freqs[self.bos] = np.inf
        freqs[self.eos] = np.inf
        freqs[self.unk] = np.inf

        freqs[self.ret] = np.inf
        freqs[self.rank] = np.inf

        freqs[self.per] = np.inf
        freqs[self.org] = np.inf
        freqs[self.misc] = np.inf
        freqs[self.loc] = np.inf
        
        self.tokens = {y: n for n, y in enumerate(sorted(freqs, key = lambda x: -freqs[x]))}

class GraphTokenizzer:
    
    is_edge = 'is' # For dealing with Named Entities
    word_token = '<WORD>'
    begin_edge = '<BED>'
    end_edge = '<EED>'
    def __init__(self, text_tokenizer) -> None:
        self.tokens = text_tokenizer.tokens
        self.cleaner = lambda x: [x] # text_tokenizer.cleaner
        self.edge_tokens = {edge_name: edge_num + len(self.tokens) for edge_num, edge_name in enumerate(['has_attribute', 'in_context_of', 'has_quantity',  'performs_action',  'receives_action', self.word_token, self.is_edge, self.begin_edge, self.end_edge])}
        self.nes_lut = {
            'per': TextTokenizer.per,
            'org': TextTokenizer.org,
            'loc': TextTokenizer.loc,
            'misc': TextTokenizer.misc,
            'word': self.word_token
        }
    
    def predict(self, graph, named_entities):
        entities = {word: self.nes_lut[entity_type] if entity_type in self.nes_lut else self.word_token for word, entity_type in named_entities} # named_entities_are [[WORD; POS], [WORD; POS], ...]
        nodes, edges = {x['id']: x['text'] for x in graph['nodes']}, graph['links']
        
        # tokens = [self.tokens[TextTokenizer.bos]]
        tokens_string = [TextTokenizer.bos]
        for edge in edges:
            source, target, label = edge['source'], edge['target'], edge['label']
            # Todo: A representation NER-Friendly that can separated named entities with maybe position or distinct tokens
            
            if nodes[source] in entities: ent_token_source = entities[nodes[source]]
            else: ent_token_source = self.nes_lut['word']
            
            cleaned_source = self.cleaner(nodes[source])[0]
            if cleaned_source in self.tokens: token_source = self.tokens[cleaned_source]
            else: token_source = self.tokens[TextTokenizer.unk]
            
            
            if nodes[target] in entities: ent_token_target = entities[nodes[target]]
            else: ent_token_target = self.nes_lut['word']
            
            cleaned_target = self.cleaner(nodes[target])[0]
            if cleaned_target in self.tokens: token_target = self.tokens[cleaned_target]
            else: token_target = self.tokens[TextTokenizer.unk]
            
            token_edge = self.edge_tokens[label]
            
            # tokens.extend([self.edge_tokens[self.begin_edge], token_source, self.edge_tokens[ent_token_source], token_edge, token_target, self.edge_tokens[ent_token_target], self.edge_tokens[self.end_edge]])
            tokens_string.extend([self.begin_edge, cleaned_source, ent_token_source, label, cleaned_target, ent_token_target, self.end_edge])
        
        return None, tokens_string + [TextTokenizer.eos]
