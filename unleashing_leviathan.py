import subprocess
import multiprocessing
import random

def run_training(params, gpu_id):
    command = f"CUDA_VISIBLE_DEVICES={gpu_id} python train_doc.py {params}"
    subprocess.run(command, shell=True)

base = '--BSIZE=100 --IMSIZE=224 --TOKEN_SIZE=256 --base_jsons=/data2/users/amolina/BOE_original/BOEv2/ --device=cuda --epochs=30  --lr=1e-05 --ntopics=256 --output_space=256 --scale=1 --test_acceptance=0.1 --text_encoder_heads=4 --text_encoder_layers=2'

topic_loss = contrastive_loss = ['HardMinerCircle', 'CLIPLoss', 'HardMinerTripletLoss', 'HardMinerCLR']
acceptance = [0.4, 0.10]
use_topic = ['image', '"None"', 'text', 'both']
model = ['ViT-B/32']
params = set()
for model_tag in model:
    for usage_of_topic in use_topic:
        for accept_rate in acceptance:
            for closs in contrastive_loss:

                if usage_of_topic != 'None':

                    for tloss in topic_loss:

                        params.add(f"{base} --acceptance={accept_rate} --closs={closs} --use_topic={usage_of_topic} --loss_function={tloss} --model_tag={model_tag}")
                else:
                    params.add(
                        f"{base} --acceptance={accept_rate} --closs={closs} --use_topic={usage_of_topic} --model_tag={model_tag}"
                    )
params = list(params)

def mp_launch(params, num_threads, thread_id, gpu):
    for idx in range(thread_id, len(params), num_threads):
        run_training(params[idx], gpu)
    return None

num_gpus = [3, 4, 5]
random.shuffle(params)
process = [multiprocessing.Process(target=mp_launch, args=(params, len(num_gpus), idx, gpu)) for idx, gpu in enumerate(num_gpus)]
[p.start() for p in process]
[p.join() for p in process]

