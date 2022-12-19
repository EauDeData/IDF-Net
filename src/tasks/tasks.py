import torch.utils.data.dataloader as dataloader
from torch.utils.tensorboard import SummaryWriter

WRITER = SummaryWriter()

class Train:

    def __init__(self, dataset, model, loss_function, tokenizer, text_prepocess, optimizer, test_task, bsize = 5, device = 'cuda'):
        
        if isinstance(dataset.tokenizer, int): 
            raise NotImplementedError(

                'For optimization reasons, ensure your dataset already contains the fitted tokenizer\n dataset.tokenizer = tokenizer will help the dataloader.'

            )
        self.loader = dataloader.DataLoader(dataset, batch_size = bsize)
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
        for n, (images, text_emb) in enumerate(self.loader):

            images, text_emb = images.to(self.device), text_emb.to(self.device)
            self.optimizer.zero_grad()
            h = self.model(images)
            loss = self.loss_f(h, text_emb)
            loss.backward()
            self.optimizer.step()

            if not (n*self.bs) % logger_freq:
                WRITER.add_scalar('Loss/train', loss.item(), epoch*n)
                print(f"Current loss: {loss.item()}")


    def run(self, epoches = 30, logger_freq = 1000):

        for epoch in range(epoches):

            self.test.epoch(500, epoch)
            self.epoch(logger_freq, epoch)

        self.test.epoch(500, epoch+1)

class Test:
    def __init__(self, dataset, model, loss_function, tokenizer, text_prepocess, optimizer, bsize = 5, device = 'cuda'):
        
        if isinstance(dataset.tokenizer, int): 
            raise NotImplementedError(

                'For optimization reasons, ensure your dataset already contains the fitted tokenizer\n dataset.tokenizer = tokenizer will help the dataloader.'

            )
        self.loader = dataloader.DataLoader(dataset, batch_size = bsize)
        self.bs = bsize
        self.model = model
        self.loss_f = loss_function
        self.tokenizer = tokenizer
        self.text_prep = text_prepocess
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
    
    def epoch(self, logger_freq, epoch):
        print(f"Testing... Epoch {epoch}")
        for n, (images, text_emb) in enumerate(self.loader):

            images, text_emb = images.to(self.device), text_emb.to(self.device)
            h = self.model(images)
            loss = self.loss_f(h, text_emb)


            if not (n*self.bs) % logger_freq:
                WRITER.add_scalar('Loss/test', loss.item(), epoch*n)
                print(f"Current loss: {loss.item()}")

    def run(self, epoches = 30, logger_freq = 500):

        for epoch in range(epoches):
            self.epoch(logger_freq, epoch)