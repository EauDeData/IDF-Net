import torch.utils.data.dataloader as dataloader
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb

from src.loss.loss import rank_correlation, raw_accuracy
from src.utils.metrics import CosineSimilarityMatrix

M = CosineSimilarityMatrix()
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
        for n, (images, text_emb) in enumerate(self.loader):

            images, text_emb = images.to(self.device), text_emb.to(self.device)
            self.optimizer.zero_grad()
            h = self.model(images)
            loss = self.loss_f(h, text_emb)
            loss.backward()
            self.optimizer.step()
            buffer += loss.item()

            if not (n*self.bs) % logger_freq:
                
                print(f"Current loss: {loss.item()}")

        WRITER.add_scalar('Loss/train', buffer/n, epoch)
        wandb.log({'train-loss': buffer / n})


    def run(self, epoches = 60, logger_freq = 1000):

        for epoch in range(epoches):
            with torch.no_grad():
                self.test.epoch(500, epoch)
                torch.save(self.model, f'output/{epoch}-{self.tokenizer.name}-{self.tokenizer.ntopics}_kullback.pkl')
            self.epoch(logger_freq, epoch)

        self.test.epoch(500, epoch+1)

class Test:
    def __init__(self, dataset, model, loss_function, tokenizer, text_prepocess, optimizer, bsize = 5, scheduler = False, device = 'cuda'):
        
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
    
    def epoch(self, logger_freq, epoch):
        print(f"Testing... Epoch {epoch}")
        buffer, pbuffer, stats_buffer = 0, 0, 0

        for n, (images, text_emb) in enumerate(self.loader):

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
            
class TrainDocAbstracts(TrainDoc):
    def __init__(self, dataset, test_set, model, bert, loss_function, tokenizer, text_prepocess, optimizer, test_task, bsize=5, device='cuda', workers=4, contrastive = None):
        super().__init__(dataset, test_set, model, bert, loss_function, tokenizer, text_prepocess, optimizer, test_task, bsize, device, workers)
        self.closs = contrastive
    
        self.loader = dataloader.DataLoader(dataset, batch_size = bsize, shuffle = True, num_workers=workers,)
        self.test_loader = dataloader.DataLoader(test_set, batch_size = bsize, shuffle = False, num_workers=workers,)   
    def epoch(self, logger_freq, epoch):
        
        print(f"Training... Epoch {epoch}")
        buffer = 0
        self.model.train()
        for n, (images, text_emb, conditional_text) in enumerate(self.loader):

            text_emb = text_emb.to(self.device)
            conditional_text = conditional_text.to(self.device)
            images = images.to(self.device)

            spotted_topics, values = self.model(images, text_emb) # TODO: Condition text properly not with the topic itself maybe
            loss = self.loss_f(spotted_topics, text_emb)
            if (not values is None) and (not self.closs is None): loss = loss + self.closs(spotted_topics, values)
            loss.backward()
            if loss != loss:

                print("Visual similarity:")
                print(M(spotted_topics, spotted_topics))

                print("Textual Similarity:")
                print(M(text_emb, text_emb))
                exit()
            self.optimizer.step()

            if not n%logger_freq: print(loss)
            buffer += loss.item()

        wandb.log({'train-loss': buffer / n})

    def test_epoch(self, logger_freq, epoch):
        
        print(f"Training... Epoch {epoch}")
        buffer, pbuffer, stats_buffer = 0, 0, 0

        self.model.eval()
        for n, (images, text_emb, conditional_text) in enumerate(self.loader):
            
            text_emb = text_emb.to(self.device)
            conditional_text = conditional_text.to(self.device)
            images = images.to(self.device)

            with torch.no_grad():
                spotted_topics, _ = self.model(images, text_emb)
                loss = self.loss_f(spotted_topics, text_emb)
                statistics, pvalues = rank_correlation(spotted_topics, text_emb,)

            if not n%logger_freq: print(loss)
            buffer += loss.item()
            pbuffer += pvalues
            stats_buffer += statistics

        wandb.log({'test-loss': buffer / n})
        
        wandb.log({'rank-corr': stats_buffer / n})
        wandb.log({'rank-corr-pvalue': pbuffer / n})

class TrainDocAbstracts(TrainDoc):
    def __init__(self, dataset, test_set, model, bert, loss_function, tokenizer, text_prepocess, optimizer, test_task, bsize=5, device='cuda', workers=4, contrastive = None):
        super().__init__(dataset, test_set, model, bert, loss_function, tokenizer, text_prepocess, optimizer, test_task, bsize, device, workers)
        self.closs = contrastive
    
        self.loader = dataloader.DataLoader(dataset, batch_size = bsize, shuffle = True, num_workers=workers,)
        self.test_loader = dataloader.DataLoader(test_set, batch_size = bsize, shuffle = False, num_workers=workers,)   
    def epoch(self, logger_freq, epoch):
        
        print(f"Training... Epoch {epoch}")
        buffer = 0
        self.model.train()
        for n, (images, text_emb, conditional_text) in enumerate(self.loader):

            text_emb = text_emb.to(self.device)
            conditional_text = conditional_text.to(self.device)
            images = images.to(self.device)

            spotted_topics, values, a = self.model(images, text_emb) # TODO: Condition text properly not with the topic itself maybe
            loss = self.loss_f(spotted_topics, text_emb)
            if (not values is None) and (not self.closs is None): loss = loss *.5 + self.closs(spotted_topics, values) *.5
            if loss != loss:
                
                print('attn weights:')
                print(a)
                print('visual tokens similarity:')
                print(M(values, values))
                print("Visual similarity:")
                print(M(spotted_topics, spotted_topics))

                print("Textual Similarity:")
                print(M(text_emb, text_emb))
                exit()
            loss.backward()
            self.optimizer.step()

            if not n%logger_freq: print('current:', loss)
            buffer += loss.item()

        wandb.log({'train-loss': buffer / n})

    def test_epoch(self, logger_freq, epoch):
        
        print(f"Testing... Epoch {epoch}")
        buffer, pbuffer, stats_buffer = 0, 0, 0

        self.model.eval()
        for n, (images, text_emb, conditional_text) in enumerate(self.loader):
            
            text_emb = text_emb.to(self.device)
            conditional_text = conditional_text.to(self.device)
            images = images.to(self.device)

            with torch.no_grad():
                spotted_topics, _, a = self.model(images, text_emb)
                loss = self.loss_f(spotted_topics, text_emb)
                statistics, pvalues = rank_correlation(spotted_topics, text_emb,)

            if not n%logger_freq: print(loss)
            buffer += loss.item()
            pbuffer += pvalues
            stats_buffer += statistics

        wandb.log({'test-loss': buffer / n})
        
        wandb.log({'rank-corr': stats_buffer / n})
        wandb.log({'rank-corr-pvalue': pbuffer / n})
