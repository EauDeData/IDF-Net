import torch.utils.data.dataloader as dataloader
import torch
import wandb

from src.loss.loss import rank_correlation, raw_accuracy
from src.utils.metrics import CosineSimilarityMatrix, get_retrieval_metrics

M = CosineSimilarityMatrix()
wandb.init(project="IDF-NET Logger")

class TrainBOE:

    def __init__(self, dataset, model, loss_function, tokenizer, text_prepocess, optimizer, test_task, bsize = 5, device = 'cuda', text_model = None, contrastive_loss = None):
        
        if isinstance(dataset.tokenizer, int): 
            raise NotImplementedError(

                'For optimization reasons, ensure your dataset already contains the fitted tokenizer\n dataset.tokenizer = tokenizer will help the dataloader.'

            )
        self.loader = dataloader.DataLoader(dataset, batch_size = bsize, shuffle = True, collate_fn = dataset.collate_boe, num_workers = 12, drop_last=True)
        self.bs = bsize
        self.model = model
        self.text_encoder = text_model
        self.contrastive = contrastive_loss

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
        self.text_encoder.train()
        for n, (images, text_emb, text) in enumerate(self.loader):
            self.optimizer.zero_grad()

            
            images = images.to(self.device)
            loss = 0
            image_embedding = self.model(images)
            scale = 1 # 0.3 if (self.text_encoder is not None) and (self.loss_f is not None) else 1
            if self.text_encoder is not None:

                topic_embedding = self.text_encoder(text.to(self.device))
                loss += self.contrastive(image_embedding, topic_embedding) * scale
            
            if self.loss_f is not None:
                loss += self.loss_f(image_embedding, text_emb.to(self.device)) * scale
            #if self.loss_f is not None and self.text_encoder is not None:
                #loss += self.loss_f(text_emb.to(self.device), topic_embedding)
            loss.backward()
            self.optimizer.step()
            buffer += loss.item()

            if not (n*self.bs) % logger_freq:
                
                print(f"Current loss: {loss.item()}")
        if not isinstance(self.test.scheduler, bool): self.test.scheduler.step(buffer / n)
        wandb.log({'train-loss': buffer / n})


    def run(self, epoches = 60, logger_freq = 1000):

        for epoch in range(epoches):
            with torch.no_grad():
                self.test.epoch(500, epoch)
                torch.save(self.model, f'output/{epoch}-{self.tokenizer.name}-{self.tokenizer.ntopics}.pkl')
            self.epoch(logger_freq, epoch)

        self.test.epoch(500, epoch+1)

class TestBOE:
    def __init__(self, dataset, model, loss_function, tokenizer, text_prepocess, optimizer, bsize = 5, scheduler = False, device = 'cuda',  text_model = None, contrastive_loss = None):
        
        if isinstance(dataset.tokenizer, int): 
            raise NotImplementedError(

                'For optimization reasons, ensure your dataset already contains the fitted tokenizer\n dataset.tokenizer = tokenizer will help the dataloader.'

            )
        self.loader = dataloader.DataLoader(dataset, batch_size = bsize, collate_fn = dataset.collate_boe, num_workers = 6, drop_last=True)
        self.bs = bsize
        self.model = model
        
        self.text_encoder = text_model
        self.contrastive = contrastive_loss

        self.loss_f = loss_function
        self.tokenizer = tokenizer
        self.text_prep = text_prepocess
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.model.to(device)
    
    def epoch(self, logger_freq, epoch):
        print(f"Testing... Epoch {epoch}")
        metrics = {
            'p-vaue topic-image': [], #
            'topic-image rank corr': [], #
            'query-image rank corr': [], #
            'topic-query rank corr': [], #
            'test-contrastive-loss': [], #
            'test-ranking-loss': [],#
            'test-loss': [], #
            'acc@1': [],
            'acc@5': [],
            'acc@10': [],
            'mAP': []

        }
        self.text_encoder.eval()
        for n, (images, text_emb, text) in enumerate(self.loader):
            with torch.no_grad():                
                images = images.to(self.device)
                image_embedding = self.model(images)
                statistics, pvalues = rank_correlation(image_embedding, text_emb.to(self.device))
                metrics['p-vaue topic-image'].append(pvalues)
                metrics['topic-image rank corr'].append(statistics)

                if self.text_encoder is not None:

                    topic_embedding = self.text_encoder(text.to(self.device))
                    loss_c = self.contrastive(image_embedding, topic_embedding).cpu().item()

                    statistics, _ = rank_correlation(image_embedding, text_emb)
                    metrics['query-image rank corr'].append(statistics)
                    metrics['test-contrastive-loss'].append(loss_c)

                    statistics, _ = rank_correlation(topic_embedding, text_emb)
                    metrics['topic-query rank corr'].append(statistics)
                    batch_metrics = get_retrieval_metrics(image_embedding, topic_embedding)

                    metrics['acc@1'].append(batch_metrics['p@1'])
                    metrics['acc@5'].append(batch_metrics['p@5'])
                    metrics['acc@10'].append(batch_metrics['p@10'])

                    metrics['mAP'].append(batch_metrics['map'])


                
                if self.loss_f is not None:
                    loss_r = self.loss_f(image_embedding, text_emb.to(self.device)).cpu().item()
                    metrics['test-ranking-loss'].append(loss_r)
                
                metrics['test-loss'].append(loss_r + loss_c)

                
            if not (n*self.bs) % logger_freq:
                
                print(f"Current loss: {metrics['test-loss'][-1]}")

        metrics = {x: sum(y)/len(y) for x,y in zip(metrics, metrics.values())}
        metrics['lr'] =  self.optimizer.param_groups[0]['lr']
        print("metrics:", metrics)
        wandb.log(metrics)
        # if not isinstance(self.scheduler, bool): self.scheduler.step(metrics['test-loss'])
        torch.save(self.model, f'output/epoch[{epoch}]-topic[{self.tokenizer.name}]-ntopics[{self.tokenizer.ntopics}]-BS[{self.bs}]-visual_encoder.pkl')
        torch.save(self.text_encoder, f'output/epoch[{epoch}]-topic[{self.tokenizer.name}]-ntopics[{self.tokenizer.ntopics}]-BS[{self.bs}]-text_encoder.pkl')

    def run(self, epoches = 30, logger_freq = 500):

        for epoch in range(epoches):
            self.epoch(logger_freq, epoch)
