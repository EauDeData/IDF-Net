import torch
import nltk 
import matplotlib.pyplot as plt
import wandb
import pickle
import numpy as np
import os
import clip
from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.text.map_text import LSALoader, TextTokenizer
from src.loss.loss import *
from src.models.models import *
from src.dataloaders.dataloaders import AbstractsDataset
from src.dataloaders.boe_dataloaders import BOEDatasetOCRd
from src.tasks.tasks_boe import TrainBOE, TestBOE
from src.tasks.evaluation import MAPEvaluation
from src.dataloaders.annoyify import Annoyifier
from src.utils.metrics import CosineSimilarityMatrix

nltk.download('stopwords')
torch.manual_seed(42)

def load_datasets(base_jsons, scale, max_imsize, mode, acceptace, test_acceptance):
    dataset = BOEDatasetOCRd(base_jsons+'train.txt', scale=scale, base_jsons=base_jsons, max_imsize=max_imsize, acceptance=acceptace, mode=mode, resize=max_imsize)
    test_data = BOEDatasetOCRd(base_jsons+'test.txt', scale=scale, base_jsons=base_jsons, max_imsize=max_imsize, mode=mode, acceptance=test_acceptance, resize=max_imsize)
    return dataset, test_data

def tokenize_text(dataset, cleaner, ntopics, model_name):
    loader = LSALoader(dataset, cleaner, ntopics=ntopics)
    loader.fit()
    text_tokenizer = TextTokenizer(cleaner)
    text_tokenizer.fit(dataset)
    pickle.dump(text_tokenizer.tokens, open(f'/data2/users/amolina/leviatan/{model_name}_tokenizer.pth', 'wb'))

    return loader, text_tokenizer

def setup_models(text_tokenizer, model_tag, device, token_size, text_encoder_heads, text_encoder_layers, output_space):
    if 'ViT' in model_tag:
        model_clip =  clip.load(model_tag, device='cpu')[0].visual
        model = TwoBranchesWrapper(model_clip, 512, output_space)
    elif 'resnet' in model_tag:
        model_clip = Resnet(1024, resnet=model_tag.split('_')[-1])
        model = TwoBranchesWrapper(model_clip, 1024, output_space)
    
    elif 'vgg' in model_tag:
        model_clip = torchvision.models.vgg19_bn()
        model = TwoBranchesWrapper(model_clip, 1000, output_space)
    elif 'convnext' in model_tag:
        model_clip = torchvision.models.convnext_tiny(weights='DEFAULT')
        model = TwoBranchesWrapper(model_clip, 1000, output_space).to(device)
    text_model = TwoBranchesWrapper(TransformerTextEncoder(len(text_tokenizer.tokens), token_size=token_size, nheads=text_encoder_heads, num_encoder_layers=text_encoder_layers), token_size, output_space).to(device)
    return model, text_model


def setup_optimizer(parameters, lr):
    optim = torch.optim.Adam(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max', threshold = 1e-3) # Maybe this should be hyperparameters
    return optim, scheduler

def get_loss_function(loss_name, batch_size):
    if loss_name == "SpearmanRankLoss":
        return SpearmanRankLoss()
    elif loss_name == "KullbackDivergenceWrapper":
        return KullbackDivergenceWrapper()
    elif loss_name == "MSERankLoss":
        return MSERankLoss()
    elif loss_name == "ContrastiveLoss":
        return ContrastiveLoss(batch_size)
    elif loss_name == "CLIPLoss":
        return CLIPLoss()
    elif loss_name == "HardMinerCircle":
        return HardMinerCircle(batch_size)
    elif loss_name == "BatchedTripletMargin":
        return BatchedTripletMargin()
    elif loss_name == "HardMinerTripletLoss":
        return HardMinerTripletLoss(batch_size)
    elif loss_name == "HardMinerCLR":
        return HardMinerCLR(batch_size)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}") 

def main(args):
    print("Use topic:", args.use_topic)
    model_name = f"use_topic_{args.use_topic}_{args.model_tag.replace('/', '-|-')}_lr_{args.lr}_loss_{args.loss_function}_closs_{args.closs}_token_{args.TOKEN_SIZE}_accept_{args.acceptance}_bsize_{args.BSIZE}_heads_{args.text_encoder_heads}_layers{args.text_encoder_layers}_output_{args.output_space}"
    if os.path.exists(f'/data2/users/amolina/leviatan/{model_name}_text_encoder.pth'): 
        print('model exists, exiting...')
        exit()
    dataset, test_data = load_datasets(args.base_jsons, args.scale, args.IMSIZE, 'text', args.acceptance, args.test_acceptance)

    print(model_name)
    print(f"Dataset loader with {len(dataset)} samples...")
    
    cleaner = StringCleanAndTrim()
    loader, text_tokenizer = tokenize_text(dataset, cleaner, args.ntopics, model_name)

    dataset.text_tokenizer = text_tokenizer
    test_data.text_tokenizer = text_tokenizer

    dataset.tokenizer = loader
    test_data.tokenizer = loader
    model, text_model = setup_models(text_tokenizer, args.model_tag, args.device, args.TOKEN_SIZE, args.text_encoder_heads, args.text_encoder_layers, args.output_space)

    parameters = list(model.parameters()) + list(text_model.parameters())
    optim, scheduler = setup_optimizer(parameters, args.lr)

    loss_function = get_loss_function(args.loss_function, args.BSIZE)
    closs = get_loss_function(args.closs, args.BSIZE)
    wandb.init(project="neoGazeta (3.0 + 1.0) - Endgame", name=model_name)
    wandb.config.update(args)

    test_task = TestBOE(test_data, model, loss_function, loader, cleaner, optim, scheduler=scheduler, device=args.device, bsize=args.BSIZE, text_model=text_model, contrastive_loss=closs, model_name=model_name)
    train_task = TrainBOE(dataset, model, loss_function, loader, cleaner, optim, test_task, device=args.device, bsize=args.BSIZE, text_model=text_model, contrastive_loss=closs, use_topic = args.use_topic)

    train_task.run(epoches=args.epochs)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Script for training and testing a model on your dataset.")
    
    # Dataset and preprocessing settings
    dataset_group = parser.add_argument_group("Dataset and Preprocessing Settings")
    dataset_group.add_argument("--base_jsons", type=str, default="/data3fast/users/amolina/BOE/", help="Base directory for JSON files")
    dataset_group.add_argument("--scale", type=int, default=1, help="Scale value for resizing")
    dataset_group.add_argument("--IMSIZE", type=int, default=224, help="Image size")
    dataset_group.add_argument("--ntopics", type=int, default=256, help="Number of topics for LSA")

    # Model and device settings
    model_group = parser.add_argument_group("Model and Device Settings")
    model_group.add_argument("--model_tag", type=str, default="ViT-B/32", help="Model tag for CLIP")
    model_group.add_argument("--device", type=str, default="cuda", help="Device for execution (cpu or cuda)")

    # Training settings
    training_group = parser.add_argument_group("Training Settings")
    training_group.add_argument("--BSIZE", type=int, default=64, help="Batch size")
    training_group.add_argument("--TOKEN_SIZE", type=int, default=64, help="Token size")
    training_group.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    training_group.add_argument("--epochs", type=int, default=600, help="Number of epochs")
    training_group.add_argument("--acceptance", type=float, default=0.5, help="Acceptance threshold for train dataloader")
    training_group.add_argument("--test_acceptance", type=float, default=0.5, help="Acceptance threshold for test dataloader")
    training_group.add_argument("--use_topic", type=str, choices=['image', 'text', 'both', 'None'])

    # Text encoder settings
    text_encoder_group = parser.add_argument_group("Text Encoder Settings")
    text_encoder_group.add_argument("--text_encoder_heads", type=int, default=2, help="Number of attention heads in text encoder")
    text_encoder_group.add_argument("--text_encoder_layers", type=int, default=2, help="Number of layers in text encoder")
    text_encoder_group.add_argument("--output_space", type=int, default=256, help="Output space dimension")
    
    # Loss function settings
    loss_group = parser.add_argument_group("Loss Function Settings")
    loss_group.add_argument("--loss_function", type=str, choices=["SpearmanRankLoss", "KullbackDivergenceWrapper", "MSERankLoss", "ContrastiveLoss", "CLIPLoss", "HardMinerCircle", "BatchedTripletMargin", "HardMinerTripletLoss", "HardMinerCLR"], default="SpearmanRankLoss", help="Loss function")
    loss_group.add_argument("--closs", type=str, choices=["SpearmanRankLoss", "KullbackDivergenceWrapper", "MSERankLoss", "ContrastiveLoss", "CLIPLoss", "HardMinerCircle", "BatchedTripletMargin", "HardMinerTripletLoss", "HardMinerCLR"], default="HardMinerCircle", help="Contrastive loss function")
    
    args = parser.parse_args()
    main(args)
