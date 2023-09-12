import subprocess
import multiprocessing
import random

def run_training(params, gpu_id):
    command = f"CUDA_VISIBLE_DEVICES={gpu_id} python train_doc.py {params}"
    subprocess.run(command, shell=True)

base = '--BSIZE=128 --IMSIZE=224 --TOKEN_SIZE=256 --base_jsons=/data3fast/users/amolina/BOE/ --device=cuda --epochs=20  --lr=1e-05 --model_tag=ViT-B/32 --ntopics=256 --output_space=256 --scale=1 --test_acceptance=0.5 --text_encoder_heads=4 --text_encoder_layers=2'
params = [
    f'{base} --acceptance=0.17 --closs=HardMinerCircle --loss_function=HardMinerCircle',
    f'{base} --acceptance=0.17 --closs=HardMinerCircle --loss_function=HardMinerCircle --topic_on_image',

    f'{base} --acceptance=0.40 --closs=HardMinerCircle --loss_function=HardMinerCircle',
    f'{base} --acceptance=0.40 --closs=HardMinerCircle --loss_function=HardMinerCircle --topic_on_image',
    
    f'{base} --acceptance=0.17 --closs=SpearmanRankLoss --loss_function=HardMinerCircle',
    f'{base} --acceptance=0.17 --closs=SpearmanRankLoss --loss_function=HardMinerCircle --topic_on_image',

    f'{base} --acceptance=0.40 --closs=SpearmanRankLoss --loss_function=HardMinerCircle',
    f'{base} --acceptance=0.40 --closs=SpearmanRankLoss --loss_function=HardMinerCircle --topic_on_image',
    
    f'{base} --acceptance=0.17 --closs=SpearmanRankLoss --loss_function=SpearmanRankLoss',
    f'{base} --acceptance=0.17 --closs=SpearmanRankLoss --loss_function=SpearmanRankLoss --topic_on_image',

    f'{base} --acceptance=0.40 --closs=SpearmanRankLoss --loss_function=SpearmanRankLoss',
    f'{base} --acceptance=0.40 --closs=SpearmanRankLoss --loss_function=SpearmanRankLoss --topic_on_image',
    
    f'{base} --acceptance=0.17 --closs=SpearmanRankLoss --loss_function=HardMinerCircle',
    f'{base} --acceptance=0.17 --closs=SpearmanRankLoss --loss_function=HardMinerCircle --topic_on_image',

    f'{base} --acceptance=0.40 --closs=SpearmanRankLoss --loss_function=HardMinerCircle',
    f'{base} --acceptance=0.40 --closs=SpearmanRankLoss --loss_function=HardMinerCircle --topic_on_image',  
    
    f'{base} --acceptance=0.40 --closs=SpearmanRankLoss --use_topic',
    f'{base} --acceptance=0.17 --closs=SpearmanRankLoss --use_topic', 
    
    f'{base} --acceptance=0.40 --closs=HardMinerCircle --use_topic',
    f'{base} --acceptance=0.17 --closs=HardMinerCircle --use_topic', 
    
    f'{base} --acceptance=0.40 --closs=HardMinerTripletLoss --use_topic',
    f'{base} --acceptance=0.17 --closs=HardMinerTripletLoss --use_topic',    
    
        
    f'{base} --acceptance=0.40 --closs=HardMinerCLR --use_topic',
    f'{base} --acceptance=0.17 --closs=HardMinerCLR --use_topic',
    
    f'{base} --acceptance=0.40 --closs=CLIPLoss --use_topic',
    f'{base} --acceptance=0.17 --closs=CLIPLoss --use_topic']

params = [
    f'{base} --acceptance=0.17 --closs=HardMinerCircle --loss_function=SpearmanRankLoss',
    f'{base} --acceptance=0.17 --closs=HardMinerCircle --loss_function=SpearmanRankLoss --topic_on_image',

    f'{base} --acceptance=0.40 --closs=HardMinerCircle --loss_function=SpearmanRankLoss',
    f'{base} --acceptance=0.40 --closs=HardMinerCircle --loss_function=SpearmanRankLoss --topic_on_image',
]

def mp_launch(params, num_threads, thread_id):
    for idx in range(thread_id, len(params), num_threads):
        run_training(params[idx], thread_id)
    return None

num_gpus = 7
random.shuffle(params)
process = [multiprocessing.Process(target=mp_launch, args=(params, num_gpus, idx)) for idx in range(num_gpus)]
[p.start() for p in process]
[p.join() for p in process]

