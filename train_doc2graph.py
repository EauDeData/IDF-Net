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


from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.text.map_text import LSALoader, TextTokenizer, GraphTokenizzer
from src.dataloaders.dataloaders import AbstractsDataset
from src.dataloaders.boe_graph_dataloaders import BOEDatasetGraph


def load_datasets(base_jsons, scale, max_imsize, acceptace, test_acceptance, tokenizer = 'graph_tokenizer'):
    processor = AutoProcessor.from_pretrained("microsoft/git-base")

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
        new_tokenizer.save_pretrained(tokenizer)
        json.dump(text_tokenizer.tokens, open(fullpath,'w'))

        print('Tokenizer trained, load again the code to use it. Yes, this is a # TODO but its getting late man')
        exit()
        
    
    dataset = BOEDatasetGraph(base_jsons+'train.txt', processor=processor, scale=scale, base_jsons=base_jsons, max_imsize=max_imsize, acceptance=acceptace, resize=max_imsize)
    test_data = BOEDatasetGraph(base_jsons+'test.txt', processor=processor, scale=scale, base_jsons=base_jsons, max_imsize=max_imsize, acceptance=test_acceptance, resize=max_imsize)
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
    dataset_group.add_argument("--acceptance", type=float, default=0.8, help="Acceptance threshold for train dataloader")
    dataset_group.add_argument("--test_acceptance", type=float, default=0.8, help="Acceptance threshold for test dataloader")
    dataset_group.add_argument('--tokenizer_path', type=str, default='new_super_graph_tokenizer')
    
    args = parser.parse_args()
    batch_size, shuffle, workers = 32, True, 0
    
    
    data, test = load_datasets(args.base_jsons, args.scale, args.IMSIZE, args.acceptance, args.test_acceptance, tokenizer=args.tokenizer_path)
    text_tokenizer = TextTokenizer(StringCleanAndTrim())
    text_tokenizer.tokens = json.load(os.path.join(args.tokenizer_path, 'proto_tokenizer.json'))
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
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
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
    print(data.prop.decode(data[0]["input_ids"][0]))
    plt.imshow(data[0]['pixel_values'])
    plt.savefig('tmp_graph.png')
    
    #### SILLY EXPERIMENT PART FOR CHECKING IF AUTOTRAIN WORKS; DELETE LATER #####
    from transformers import ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel
    from transformers import VisionEncoderDecoderModel , ViTFeatureExtractor
    
    ENCODER = "google/vit-base-patch16-224"
    DECODER = "gpt2"
    
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(ENCODER, DECODER)
    model.config.decoder_start_token_id = data.prop.tokenizer.cls_token_id
    model.config.pad_token_id =  data.prop.tokenizer.pad_token_id
    
    
    training_args = Seq2SeqTrainingArguments(
        output_dir='VIT_large_gpt2',
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        predict_with_generate=True,
        evaluation_strategy="epoch",
        do_train=True,
        do_eval=True,
        logging_steps=1024,  
        save_steps=2048, 
        warmup_steps=1024,  
        learning_rate = 5e-5,
        #max_steps=1500, # delete for full training
        num_train_epochs = 50, #TRAIN_EPOCHS
        overwrite_output_dir=True,
        save_total_limit=1,
    )
    # instantiate trainer
    trainer = Seq2SeqTrainer(
        tokenizer=ViTFeatureExtractor.from_pretrained(ENCODER),
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=data,
        eval_dataset=test,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model('output/doc2graph_gpt_vit')


'''

    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for epoch in range(50):
        print("Epoch:", epoch)
        
        print('testing-...')
        model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(train_loader):
                input_ids = batch.pop("input_ids").to(device)
                pixel_values = batch.pop("pixel_values").to(device)

                outputs = model(input_ids=input_ids,
                                pixel_values=pixel_values,
                                labels=input_ids)
                
                loss = outputs.loss

                print("Loss:", loss.item())
                print("Out:", outputs) 
        
        model.train()
 
        print('training...')
        for idx, batch in enumerate(train_loader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=input_ids)
            
            loss = outputs.loss

            print("Loss:", loss.item())

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            



   
 '''   
    