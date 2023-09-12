import torch
import nltk 
import matplotlib.pyplot as plt
import wandb
import pickle
import numpy as np
import clip
from transformers import AutoProcessor
from torch.utils.data import DataLoader
import os
import json
from transformers import Seq2SeqTrainer ,Seq2SeqTrainingArguments
from transformers import AutoTokenizer ,  GPT2Config , default_data_collator
from transformers import GPT2Tokenizer

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.text.map_text import LSALoader, TextTokenizer, GraphTokenizzer
from src.dataloaders.dataloaders import AbstractsDataset
from src.dataloaders.boe_graph_dataloaders import BOEDatasetGraph


def load_datasets(base_jsons, scale, max_imsize, acceptace, test_acceptance, model_name, imsize = None, tokenizer = 'graph_tokenizer', train_tokenizer = False):
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    # processor.tokenizer =  AutoTokenizer.from_pretrained(model_name)
    print('train tokenizer:', train_tokenizer)
    if train_tokenizer:
        if os.path.exists(tokenizer):
            print('Loading Tokenizer')
            processor.tokenizer = processor.tokenizer.from_pretrained(tokenizer)
            
        else:
            print(f'Tokenizing graphs, {tokenizer} does not exist')
            text_tokenizer = TextTokenizer(StringCleanAndTrim())
            
            dataset = BOEDatasetGraph(base_jsons+'train.txt',processor=processor, scale=scale, base_jsons=base_jsons, max_imsize=max_imsize, acceptance=acceptace, resize=max_imsize)
            print('Processing text for graph parsing')

            text_tokenizer.fit(dataset)
            fullpath = os.path.join(tokenizer, 'proto_tokenizer.json')
            
            graph_tokenizer = GraphTokenizzer(text_tokenizer)
            
            print('Training New Tokenizer from HF')
            new_tokenizer = processor.tokenizer.train_new_from_iterator((' '.join(graph_tokenizer.predict(x['graph'], x['NEs'])[1]) for x in dataset.data), len(graph_tokenizer.edge_tokens) + len(graph_tokenizer.tokens))
            special_tokens = {'additional_special_tokens':[*list(graph_tokenizer.edge_tokens.keys()), text_tokenizer.bos, text_tokenizer.unk, text_tokenizer.eos, *list(graph_tokenizer.nes_lut.values())] + processor.tokenizer.all_special_tokens}
            
            new_tokenizer.add_special_tokens(special_tokens)

            new_tokenizer.save_pretrained(tokenizer)
            json.dump(text_tokenizer.tokens, open(fullpath,'w'))

            print('Tokenizer trained, load again the code to use it. Yes, this is a # TODO but its getting late man')
            exit()
    else:
        
        print('loading gpt2 tokenizer')
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        text_tokenizer = TextTokenizer(StringCleanAndTrim())
        text_tokenizer.tokens = {}
        graph_tokenizer = GraphTokenizzer(text_tokenizer)
        
        special_tokens = {'additional_special_tokens':[*list(graph_tokenizer.edge_tokens.keys()), text_tokenizer.bos, text_tokenizer.unk, text_tokenizer.eos, *list(graph_tokenizer.nes_lut.values())] + processor.tokenizer.all_special_tokens}
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.add_special_tokens(special_tokens)
        processor.tokenizer = tokenizer
            
    
    dataset = BOEDatasetGraph(base_jsons+'train.txt', processor=processor, scale=scale, base_jsons=base_jsons, resize=imsize, max_imsize=max_imsize,min_height=512, min_width=512, acceptance=acceptace)
    test_data = BOEDatasetGraph(base_jsons+'test.txt', processor=processor, scale=scale, base_jsons=base_jsons, resize=imsize, max_imsize=max_imsize, acceptance=test_acceptance, min_height=512, min_width=512)
    return dataset, test_data



if __name__ == "__main__":
    print('HOLAAAA')
    import argparse

    parser = argparse.ArgumentParser(description="Script for training and testing a model on your dataset.")
    
    # Dataset and preprocessing settings
    dataset_group = parser.add_argument_group("Dataset and Preprocessing Settings")
    dataset_group.add_argument("--base_jsons", type=str, default="/data3fast/users/amolina/BOE/", help="Base directory for JSON files")
    dataset_group.add_argument("--scale", type=int, default=1, help="Scale value for resizing")
    dataset_group.add_argument("--IMSIZE", type=int, default=224, help="Image size")
    dataset_group.add_argument("--acceptance", type=float, default=0.4, help="Acceptance threshold for train dataloader")
    dataset_group.add_argument("--test_acceptance", type=float, default=0.4, help="Acceptance threshold for test dataloader")
    dataset_group.add_argument('--tokenizer_path', type=str, default='new_super_graph_tokenizer')
    dataset_group.add_argument('--batch_size', type=int, default=8)
    dataset_group.add_argument('--train_tokenizer', type=bool, default=False)
    
    
    model_group = parser.add_argument_group("Trainer and Model Parameters")
    dataset_group.add_argument('--visual_encoder', type=str, default="google/vit-base-patch16-224")
    dataset_group.add_argument('--textual_encoder', type=str, default="gpt2")
    dataset_group.add_argument('--epoches', type=int, default=200)
    dataset_group.add_argument('--lr', type=float, default=1e-5)
    dataset_group.add_argument('--visual_depth', type=int, default=6)
    dataset_group.add_argument('--visual_width', type=int, default=6)
    
    dataset_group.add_argument('--textual_depth', type=int, default=6)
    dataset_group.add_argument('--textual_width', type=int, default=6)    
    
    dataset_group.add_argument('--dropout', type=float, default=0.1)
    
    
    
    
    args = parser.parse_args()
    batch_size, shuffle, workers = args.batch_size, True, 12
    
    
    data, test = load_datasets(args.base_jsons, args.scale, args.IMSIZE, args.acceptance, args.test_acceptance, args.textual_encoder, imsize=args.IMSIZE,tokenizer=args.tokenizer_path, train_tokenizer=True)
    text_tokenizer = TextTokenizer(StringCleanAndTrim())
    text_tokenizer.tokens = json.load(open(os.path.join(args.tokenizer_path, 'proto_tokenizer.json')))
    tokenizer = data.prop.tokenizer


    train_loader, test_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=workers), DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=workers)
    
    graph_tokenizer = GraphTokenizzer(text_tokenizer)
    data.graph_tokenizer = graph_tokenizer
    test.graph_tokenizer = graph_tokenizer
    
    import datasets
    rouge = datasets.load_metric("rouge")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # all unnecessary tokens are removed
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id # WTF IS THAT, CHECK IT OUT BC ITS DANGEROUS
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }
    
    print('batch:')
    print(data[0])
    print('decoded:')
    print(data.prop.decode(data[0]["input_ids"]))
    #### SILLY EXPERIMENT PART FOR CHECKING IF AUTOTRAIN WORKS; DELETE LATER #####
    import torch
    from transformers import ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
    from transformers import VisionEncoderDecoderModel , ViTFeatureExtractor
    import wandb
    from tqdm import tqdm
    from transformers import get_scheduler, get_linear_schedule_with_warmup
    import transformers
    from transformers import ViTConfig, ViTModel
    
    
    ENCODER = args.visual_encoder
    DECODER = args.textual_encoder
    outname = f"doc2graph_{ENCODER.split('/')[-1]+'_tmpname_'}_{DECODER.split('/')[-1]+'_tmpname_'}_epoches_{args.epoches}_ACCEPTANCE_{args.acceptance}_{args.lr}"
    
    configuration = ViTConfig(image_size=args.IMSIZE, num_hidden_layers=args.visual_depth, num_attention_heads=args.visual_width, hidden_dropout_prob=args.dropout  )
    decoder_config = GPT2Config(embd_pdrop = args.dropout, n_layer = args.textual_depth, n_head = args.textual_width, vocab_size = len(data.prop.tokenizer))
    
    conf = VisionEncoderDecoderConfig.from_encoder_decoder_configs(configuration, decoder_config)
    model = VisionEncoderDecoderModel(config=conf)
    
    model.config.decoder_start_token_id = data.prop.tokenizer.cls_token_id
    model.config.pad_token_id = data.prop.tokenizer.pad_token_id
    
    
    # model.decoder.decoder_start_token_id = data.prop.tokenizer.bos_token_id
    # model.decoder.pad_token_id =  data.prop.tokenizer.pad_token_id
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    from transformers import TrainingArguments, Trainer

    run = wandb.init(project = 'doc2graph')
    # scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 1000, args.epoches * len(train_loader), num_cycles = int(args.epoches * len(train_loader) / 30000))


    for epoch in range(args.epoches):
        model.save_pretrained(f"output/{outname}")
        print("Epoch:", epoch)
        model.train()
 
        buffer = 0
        print('training...')
        for idx, batch in tqdm(enumerate(train_loader), desc = f' epoch {epoch} training...'):
            
            attn = batch.pop('attention_mask').to(device)
            input_ids = batch.pop("input_ids").to(device) * attn
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model(pixel_values=pixel_values,
                            labels=input_ids)
            
            loss = outputs.loss

            loss.backward()
            buffer += loss.item()

            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            run.log({
                'lr': optimizer.param_groups[0]['lr']
            })
        run.log({
            'train-loss': buffer / (idx+1)
        })
            
        
        print('testing-...')
        model.eval()
        buffer = 0
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(test_loader), desc = f'epoch {epoch} testing...'):

                attn = batch.pop('attention_mask').to(device)
                input_ids = batch.pop("input_ids").to(device) * attn
                
                pixel_values = batch.pop("pixel_values").to(device)

                outputs = model(pixel_values=pixel_values, labels=input_ids)

                
                loss = outputs.loss
                buffer += loss
            run.log({
            'test-loss': buffer / (idx + 1),
            })

        generated_ids = model.generate(pixel_values)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print(generated_text)
 

    



'''
    training_args = TrainingArguments(
        output_dir="./output",  # Output directory for checkpoints and logs
        overwrite_output_dir=True,  # Overwrite the content of the output directory
        num_train_epochs=args.epoches,  # Number of training epochs
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=100,  # Save checkpoint every specified number of steps
        logging_dir="./logs",  # Directory for storing logs
        logging_steps=10,  # Log metrics every specified number of steps
        report_to='wandb',
        evaluation_strategy='no',
        auto_find_batch_size=False,
        warmup_steps=100,
        
        
        
    )
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data,  # Pass the dataset directly to the Trainer
    eval_dataset=test,
    data_collator=default_data_collator,  # You can change this as needed
    compute_metrics=compute_metrics,  # Your compute_metrics function
    optimizers = (optimizer, transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=args.epoches * (len(data) / batch_size)))
    )

    trainer.train()
'''
