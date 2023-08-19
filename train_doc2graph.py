import torch
import nltk 
import matplotlib.pyplot as plt
import wandb
import pickle
import numpy as np
import clip
from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.text.map_text import LSALoader, TextTokenizer, GraphTokenizzer
from src.dataloaders.dataloaders import AbstractsDataset
from src.dataloaders.boe_graph_dataloaders import BOEDatasetGraph

def load_datasets(base_jsons, scale, max_imsize, acceptace, test_acceptance):
    dataset = BOEDatasetGraph(base_jsons+'train.txt', scale=scale, base_jsons=base_jsons, max_imsize=max_imsize, acceptance=acceptace, resize=max_imsize)
    test_data = BOEDatasetGraph(base_jsons+'test.txt', scale=scale, base_jsons=base_jsons, max_imsize=max_imsize, acceptance=test_acceptance, resize=max_imsize)
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
    
    args = parser.parse_args()
    
    data, test = load_datasets(args.base_jsons, args.scale, args.IMSIZE, args.acceptance, args.test_acceptance)
    text_tokenizer = TextTokenizer(StringCleanAndTrim())
    text_tokenizer.fit(data)
    
    graph_tokenizer = GraphTokenizzer(text_tokenizer)
    data.graph_tokenizer = graph_tokenizer
    test.graph_tokenizer = graph_tokenizer
    print(data[0])
    plt.imshow(data[0]['pixel_values'])
    plt.savefig('tmp_graph.png')
    #### SILLY EXPERIMENT PART FOR CHECKING IF AUTOTRAIN WORKS; DELETE LATER #####
    from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel
    from transformers import ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel

    config_encoder = ViTConfig()
    config_decoder = BertConfig()
    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
    model = VisionEncoderDecoderModel(config=config)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    from transformers import Trainer, TrainingArguments
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model=model.to(device)
    model.train()
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=30,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
    )



    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=data,         # training dataset
        eval_dataset=test            # evaluation dataset
    )
    trainer.train()