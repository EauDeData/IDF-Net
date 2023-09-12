import os
import random
import subprocess

all_models = list(set([os.path.splitext(a)[0].replace('_visual_encoder', '') for a in  os.listdir('/data3fast/users/amolina/leviatan/') if 'visual_encoder' in a]))
random.shuffle(all_models)

command = "CUDA_VISIBLE_DEVICES=9 python vis.py --name '{}'"
for model in all_models:
    subprocess.run(command.format(model), shell=True)