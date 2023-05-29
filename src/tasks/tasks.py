import torch.utils.data.dataloader as dataloader
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
import os

from src.loss.loss import rank_correlation, raw_accuracy, CosineSimilarityMatrix

wandb.init(project="IDF-NET Logger")
WRITER = SummaryWriter()

class Train:

    def __init__(self, dataset, model, loss_function, tokenizer, text_prepocess, optimizer, test_task, bsize = 5, device = 'cuda',):
        
        if isinstance(dataset.tokenizer, int): 
            raise NotImplementedError(

                'For optimization reasons, ensure your dataset already contains the fitted tokenizer\n dataset.tokenizer = tokenizer will help the dataloader.'

            )
        self.loader = dataloader.DataLoader(dataset, batch_size = bsize, shuffle = True)
        self.bs = bsize
        self.model = model
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
        for n, (images, text_emb, text) in enumerate(self.loader):

            # print(text)

            images, text_emb = images.to(self.device), text_emb.to(self.device)
            self.optimizer.zero_grad()
            h = self.model(images)
            loss = self.loss_f(h, text_emb)
            assert loss==loss, "me quiero matar"
            loss.backward()
            self.optimizer.step()
            buffer += loss.item()

            if not (n*self.bs) % logger_freq:
                
                print(f"Current loss: {loss.item()}")

        WRITER.add_scalar('Loss/train', buffer/n, epoch)
        wandb.log({'train-loss': buffer / n})


    def run(self, epoches = 60, logger_freq = 1000):

        for epoch in range(epoches):

            self.epoch(logger_freq, epoch)
            with torch.no_grad():
                self.test.epoch(500, epoch)
        self.test.epoch(500, epoch+1)

class Test:
    def __init__(self, dataset, model, loss_function, tokenizer, text_prepocess, optimizer, bsize = 5, scheduler = False, save = True, device = 'cuda'):
        
        if isinstance(dataset.tokenizer, int): 
            raise NotImplementedError(

                'For optimization reasons, ensure your dataset already contains the fitted tokenizer\n dataset.tokenizer = tokenizer will help the dataloader.'

            )
        self.loader = dataloader.DataLoader(dataset, batch_size = bsize, shuffle = True)
        self.bs = bsize
        self.model = model
        self.loss_f = loss_function
        self.tokenizer = tokenizer
        self.text_prep = text_prepocess
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.model.to(device)
        self.save = save
    
    def epoch(self, logger_freq, epoch):
        print(f"Testing... Epoch {epoch}")
        buffer, pbuffer, stats_buffer = 0, 0, 0

        if self.save:
            if not os.path.exists('./output/'): os.mkdir('./output')
            if not os.path.exists('./output/models/'):  os.mkdir('./output/models/')
            torch.save(self.model.state_dict(), f'./output/models/{epoch}.pth')

        for n, (images, text_emb, _) in enumerate(self.loader):

            images, text_emb = images.to(self.device), text_emb.to(self.device)
            h = self.model(images)
            loss = self.loss_f(h, text_emb)
            buffer += loss.item()

            statistics, pvalues = rank_correlation(h, text_emb,)
            stats_buffer += statistics
            pbuffer += pvalues

            if not (n*self.bs) % logger_freq:
                
                print(f"Current loss: {loss.item()}")
        WRITER.add_scalar('Loss/test', buffer/n, epoch)

        wandb.log({'test-loss': buffer / n})
        wandb.log({'rank-corr': stats_buffer / n})
        wandb.log({'rank-corr-pvalue': pbuffer / n})
        if not isinstance(self.scheduler, bool): self.scheduler.step(buffer / n)

    def run(self, epoches = 30, logger_freq = 500):
        
        for epoch in range(epoches):
            self.epoch(logger_freq, epoch)
