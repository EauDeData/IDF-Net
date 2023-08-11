import os
import annoy
import torch
import warnings

from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class Annoyer:
    # High performance approaximate nearest neighbors - agnostic wrapper
    # Find implementation and documentation on https://github.com/spotify/annoy

    def __init__(self, model_visual, model_textual, cleaner, tokenizer, dataset, emb_size=None, ntopics = None, distance='angular', experiment_name='resnet_base', out_dir='output/', device='cuda') -> None:
        assert not (emb_size is None) and isinstance(emb_size, int),\
            f'When using Annoyer KNN emb_size must be an int. Set as None for common interface. Found: {type(emb_size)}'

        self.model_visual = model_visual
        self.model_textual = model_textual

        self.cleaner = cleaner
        self.tokenizer = tokenizer

        # FIXME: Dataloader assumes 1 - Batch Size
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1, num_workers = 6, shuffle = False, collate_fn = dataset.collate_boe)
        self.device = device

        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(
            out_dir, f'index.ann')

        self.vistrees = annoy.AnnoyIndex(emb_size, distance)
        self.texttrees = annoy.AnnoyIndex(emb_size, distance)
        self.topictrees = annoy.AnnoyIndex(ntopics, distance)
        self.trees = [
            self.vistrees,
            self.texttrees,
            self.topictrees
        ]

        self.state_variables = {
            'built': False,
        }
    def add_to_trees(self, *args, idx = None):
        for tree, arg in enumerate(args):
            self.trees[tree].add_item(idx, arg)
    
    def build(self, n):
        for tree in self.trees: tree.build(n)

    def save(self, *args):
        for tree, arg in enumerate(args):
            self.trees[tree].save(self.path+arg)

    def fit(self):
        if self.state_variables['built']:
            raise AssertionError('Cannot fit a built Annoy')
        else:
            self.state_variables['built'] = True
        
        for idx, (images, text_emb, text) in enumerate(self.dataloader):
            print(
                f'Building KNN... {idx} / {len(self.dataloader)}\t', end='\r')

            with torch.no_grad():
                vis_emb  = self.model_visual(images.cuda())[0].squeeze( )  # Ensure batch_size = 1
                text_emb_extracted  = self.model_textual(text.cuda())[0].squeeze( )  # Ensure batch_size = 1
            self.add_to_trees(vis_emb, text_emb_extracted, text_emb.squeeze( ), idx = idx)

        self.build(10)  # 10 trees
        self.save('visual', 'textual', 'topic')

    def load(self, *args):
        if self.state_variables['built']:
            raise AssertionError('Cannot load an already built Annoy')
        else:
            self.state_variables['built'] = True

        for tree, arg in enumerate(args):
            self.trees[tree].load(self.path+arg)

    def retrieve_by_idx(self, idx, n=50, idx_tree = 0, **kwargs):

        return self.trees[idx_tree].get_nns_by_item(idx, n, **kwargs)

    def retrieve_by_vector(self, vector, n=50, idx_tree = 0, **kwargs):
        return self.trees[idx_tree].get_nns_by_vector(vector, n, **kwargs)
