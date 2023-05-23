import torch.utils.data.dataloader as dataloader
import torch
import wandb
import os
import sys
from src.loss.loss import rank_correlation, raw_accuracy


def print_total_gradient_shape(model):
    total_gradient = 0
    for param in model.parameters():
        if param.grad is not None:
            total_gradient += sys.getsizeof(param.grad.numel())
    print("Total Gradient Shape:", total_gradient)

class TrainDoc:

    def __init__(self, dataset, test_set, model, bert, loss_function, tokenizer, text_prepocess, optimizer, test_task, bsize = 5, device = 'cuda', workers = 16):
        
        if isinstance(dataset.tokenizer, int): 
            raise NotImplementedError(

                'For optimization reasons, ensure your dataset already contains the fitted tokenizer\n dataset.tokenizer = tokenizer will help the dataloader.'

            )
        self.loader = dataloader.DataLoader(dataset, batch_size = bsize, shuffle = True, num_workers=workers, collate_fn=dataset.collate_boe)
        self.test_loader = dataloader.DataLoader(test_set, batch_size = bsize, shuffle = False, num_workers=workers, collate_fn=dataset.collate_boe)
        self.bert = bert
        self.bs = bsize
        self.model = model.to(device)
        self.loss_f = loss_function
        self.tokenizer = tokenizer
        self.text_prep = text_prepocess
        self.optimizer = optimizer
        self.test = test_task
        self.device = device
        self.model.to(self.device)
    
    def epoch(self, logger_freq, epoch):
        
        print(f"Training... Epoch {epoch}")
        buffer = 0
        self.model.train()
        text_embs = []
        loss = None
        nans = 0
        for n, (images, mask, text_emb, text, conditional_text) in enumerate(self.loader):

            text_emb = text_emb.to(self.device)
            conditional_text = conditional_text.to(self.device)

            text_embs.append(text_emb)

            images, mask, text_emb = images.to(self.device), mask.to(self.device), text_emb.to(self.device)
            try:
                h = self.model(images, mask, conditional_text)
            except:
                print(images.shape)
                exit()


            if not h is None:
                
                embs = torch.cat(text_embs, dim = 0)
                loss = self.loss_f(h, embs)
                loss.backward() # prevent NaN for constant arrays

                if loss == loss:

                    self.optimizer.step()
                    buffer += loss.item()
                    text_embs = []

                else:
                    nans += 1

                self.optimizer.zero_grad()
                self.model.zero_grad()
                text_embs = []

            if not (n % logger_freq): print(f"Current loss: {loss}, NaNs: {nans}")

        wandb.log({'train-loss': buffer / n})
    
    def test_epoch(self, logger_freq, epoch):
        
        print(f"Testing... Epoch {epoch}")
        buffer, pbuffer, stats_buffer = 0, 0, 0
        self.model.eval()
        text_embs = []
        loss = None
        for n, (images, mask, text_emb, text, conditional_text) in enumerate(self.loader):

            text_emb = text_emb.to(self.device)
            conditional_text = conditional_text.to(self.device)

            text_embs.append(text_emb)

            images, mask, text_emb = images.to(self.device), mask.to(self.device), text_emb.to(self.device)
            with torch.no_grad():

                h = self.model(images, mask, conditional_text)
                if not h is None:
                    embs = torch.cat(text_embs, dim = 0)

                    loss = self.loss_f(h, embs)
                    text_embs = []

                    self.optimizer.step()
                    buffer += loss.item()

                    
                    statistics, pvalues = rank_correlation(h, embs,)
                    stats_buffer += statistics
                    pbuffer += pvalues

            if not (n*self.bs) % logger_freq:
                
                print(f"Current loss: {loss}")

        wandb.log({'test-loss': buffer / n})
        wandb.log({'rank-corr': stats_buffer / n})
        wandb.log({'rank-corr-pvalue': pbuffer / n})
    
    def train(self, epoches):
        for epoch in range(epoches):
            self.epoch(36, epoch)
            self.test_epoch(36, epoch)
            
