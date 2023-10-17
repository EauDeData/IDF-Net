import torch.utils.data.dataloader as dataloader
import torch
import wandb

from src.loss.loss import rank_correlation, raw_accuracy
from src.utils.metrics import CosineSimilarityMatrix, get_retrieval_metrics

M = CosineSimilarityMatrix()

class TrainBOE:

    def __init__(self, dataset, model, loss_function, tokenizer, text_prepocess, optimizer, test_task, use_topic =True, bsize = 5, device = 'cuda', topic_on_image = None, text_model = None, contrastive_loss = None):
        
        if isinstance(dataset.tokenizer, int): 
            raise NotImplementedError(

                'For optimization reasons, ensure your dataset already contains the fitted tokenizer\n dataset.tokenizer = tokenizer will help the dataloader.'

            )
        self.loader = dataloader.DataLoader(dataset, batch_size = bsize, shuffle = True, collate_fn = dataset.collate_boe, num_workers = 12, pin_memory=False, drop_last=True)
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
        self.use_topic = use_topic
        self.topic_on_image = topic_on_image
        self.model.to(self.device)
    
    def epoch(self, logger_freq, epoch):
        
        print(f"Training... Epoch {epoch}")
        buffer = 0
        self.model.train()
        self.text_encoder.train()
        for n, (images, topic_lda, query_tokens) in enumerate(self.loader):
            self.optimizer.zero_grad()

            # print(f"Images: {images.shape}, topic: {topic_lda.shape}, text: {query_tokens.shape}")
            images = images.to(self.device)
            loss = 0
            image_embedding, _ = self.model(images)
            topic_embedding, _ = self.text_encoder(query_tokens.to(self.device))

            if self.use_topic != 'None':
                if self.use_topic != 'both': scale = 0.5
                else: scale = 0.25
            else: scale = 1
            
            loss += self.contrastive(image_embedding, topic_embedding) * (scale if self.use_topic != 'both' else 2 * scale)
            if self.use_topic == 'text':
                loss += self.loss_f(topic_lda.to(self.device), topic_embedding) * scale
            
            elif self.use_topic == 'image':
                loss += self.loss_f(topic_lda.to(self.device), image_embedding) * scale

            else:
                loss += self.loss_f(topic_lda.to(self.device), image_embedding) * scale + self.loss_f(topic_lda.to(self.device), topic_embedding) * scale

            loss.backward()
            self.optimizer.step()
            buffer += loss.item()

            if not (n*self.bs) % logger_freq:
                
                print(f"Current loss: {loss.item()}")
        # if not isinstance(self.test.scheduler, bool): self.test.scheduler.step(buffer / n)
        wandb.log({'train-loss': buffer / n})


    def run(self, epoches = 60, logger_freq = 1000):

        for epoch in range(epoches):
            with torch.no_grad():
                self.test.epoch(500, epoch)
            self.epoch(logger_freq, epoch)

        self.test.epoch(500, epoch+1)

class TestBOE:
    def __init__(self, dataset, model, loss_function, tokenizer, text_prepocess, optimizer, bsize = 5, scheduler = False, device = 'cuda',  text_model = None, contrastive_loss = None, model_name = 'model.pkl'):
        
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
        self.model_name = model_name
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
            'text2img acc@1': [],
            'text2img acc@5': [],
            'text2img acc@10': [],
            'text2img mAP': [],

            'img2text acc@1': [],
            'img2text acc@5': [],
            'img2text acc@10': [],
            'img2text mAP': []
        }
        self.text_encoder.eval()
        self.model.eval()
        for n, (images, text_emb, text) in enumerate(self.loader):
            with torch.no_grad():                
                images = images.to(self.device)
                image_embedding, rank_img_emb = self.model(images)
                statistics, pvalues = rank_correlation(image_embedding, text_emb.to(self.device))
                metrics['p-vaue topic-image'].append(pvalues)
                metrics['topic-image rank corr'].append(statistics)

                if self.text_encoder is not None:

                    retrieval_embedding, rank_embedding = self.text_encoder(text.to(self.device))
                    loss_c = self.contrastive(image_embedding, retrieval_embedding).cpu().item()

                    statistics, _ = rank_correlation(image_embedding, text_emb)
                    metrics['query-image rank corr'].append(statistics)
                    metrics['test-contrastive-loss'].append(loss_c)

                    statistics, _ = rank_correlation(retrieval_embedding, text_emb)
                    metrics['topic-query rank corr'].append(statistics)

                    ### RETRIEVAL ###
                    batch_metrics = get_retrieval_metrics(image_embedding, retrieval_embedding)

                    metrics['text2img acc@1'].append(batch_metrics['p@1'])
                    metrics['text2img acc@5'].append(batch_metrics['p@5'])
                    metrics['text2img acc@10'].append(batch_metrics['p@10'])

                    metrics['text2img mAP'].append(batch_metrics['map'])

                    #### CAPTIONNING ###
                    batch_metrics = get_retrieval_metrics(image_embedding, retrieval_embedding, img2text = True)

                    metrics['img2text acc@1'].append(batch_metrics['p@1'])
                    metrics['img2text acc@5'].append(batch_metrics['p@5'])
                    metrics['img2text acc@10'].append(batch_metrics['p@10'])

                    metrics['img2text mAP'].append(batch_metrics['map'])
                
                if self.loss_f is not None:
                    loss_r = self.loss_f(image_embedding, text_emb.to(self.device)).cpu().item()
                    metrics['test-ranking-loss'].append(loss_r)
                
                metrics['test-loss'].append(loss_r + loss_c)

                
            if not (n*self.bs) % logger_freq:
                
                print(f"Current loss: {metrics['test-loss'][-1]}")

        metrics = {x: sum(y)/len(y) for x,y in zip(metrics, metrics.values())}
        metrics['lr'] =  self.optimizer.param_groups[0]['lr']
        print("metrics:", metrics)
        if not isinstance(self.scheduler, bool): self.scheduler.step(metrics['img2text acc@1'])

        wandb.log(metrics)
        torch.save(self.model.state_dict(), f'/data2/users/amolina/leviatan/{self.model_name}_visual_encoder.pth')
        torch.save(self.text_encoder.state_dict(), f'/data2/users/amolina/leviatan/{self.model_name}_text_encoder.pth')

    def run(self, epoches = 30, logger_freq = 500):

        for epoch in range(epoches):
            self.epoch(logger_freq, epoch)
