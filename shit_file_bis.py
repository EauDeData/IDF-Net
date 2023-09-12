from transformers import AutoTokenizer
from transformers import AutoImageProcessor
from transformers import VisionEncoderDecoderModel , ViTFeatureExtractor
from src.dataloaders.boe_graph_dataloaders import BOEDatasetGraph
from transformers import AutoProcessor
from src.text.map_text import LSALoader, TextTokenizer, GraphTokenizzer
from src.text.preprocess import StringCleanAndTrim, StringCleaner

import os
import torch
import json
import cv2
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("./proto_tokenizer_bert")
processor = AutoProcessor.from_pretrained("microsoft/git-base")
processor.tokenizer = tokenizer

model = VisionEncoderDecoderModel.from_pretrained('./output/doc2graph_vit-base-patch16-224_tmpname__gpt2_tmpname__epoches_200_ACCEPTANCE_0.5_1e-05/')
model.eval()

base_jsons = '/data3fast/users/amolina/BOE/'
test_data = BOEDatasetGraph(base_jsons+'test.txt', processor=processor, scale=1, base_jsons=base_jsons, max_imsize=np.inf, min_height=512, min_width=512, acceptance=0.5, resize=224)


text_tokenizer = TextTokenizer(StringCleanAndTrim())
text_tokenizer.tokens = json.load(open(os.path.join("./new_super_tokenizer", 'proto_tokenizer.json')))
    
graph_tokenizer = GraphTokenizzer(text_tokenizer)
test_data.graph_tokenizer = graph_tokenizer

idx = 505
sample = test_data[idx]
image = sample.pop('pixel_values').unsqueeze(0)

datapoint = test_data.data[idx]
page = datapoint['topic_gt']["page"]
segment = datapoint['topic_gt']["idx_segment"]
original_image = np.load(datapoint['root'])[page]
x, y, x2, y2 = datapoint['pages'][page][segment]['bbox']
original_image = original_image[y:y2, x:x2]
cv2.imwrite('original_for_graph_test.png', original_image)

page = datapoint['topic_gt']["page"]
print('Query:', datapoint['query'])

numpy_image = image.squeeze().mean(0).cpu().detach().numpy()
import matplotlib.pyplot as plt
plt.imshow(numpy_image)
plt.savefig('tmp.png')
test = open('test_graph.txt', 'w')
test.write('GT:\n')
test.write(' '.join([t for t in tokenizer.batch_decode(sample['input_ids'], skip_special_tokens=False) if t!='[PAD]']))
print(' '.join([t for t in tokenizer.batch_decode(sample['input_ids'], skip_special_tokens=False) if t!='[PAD]']))
with torch.no_grad():
    generated_text = tokenizer.batch_decode(model.generate(image, max_length = 50), skip_special_tokens=False)
    
test.write('\nprediction:\n')
test.write(' '.join(generated_text))
test.close()
print('======================================')
train_data = BOEDatasetGraph(base_jsons+'train.txt', processor=processor, scale=1, base_jsons=base_jsons, max_imsize=np.inf, min_height=512, min_width=512, acceptance=0.5, resize=224)


idx = 0
sample = train_data[idx]
image = sample.pop('pixel_values').unsqueeze(0)
test = open('train_graph.txt', 'w')

datapoint = train_data.data[idx]
page = datapoint['topic_gt']["page"]
segment = datapoint['topic_gt']["idx_segment"]
original_image = np.load(datapoint['root'])[page]
x, y, x2, y2 = datapoint['pages'][page][segment]['bbox']
original_image = original_image[y:y2, x:x2]
cv2.imwrite('original_for_graph_train.png', original_image)

numpy_image = image.squeeze().mean(0).cpu().detach().numpy()
import matplotlib.pyplot as plt
plt.imshow(numpy_image)
plt.savefig('tmp.png')
test.write('GT:\n')
test.write(' '.join([t for t in tokenizer.batch_decode(sample['input_ids'], skip_special_tokens=False) if t!='[PAD]']))
with torch.no_grad():
    generated_text = tokenizer.batch_decode(model.generate(image, max_length = 50), skip_special_tokens=False)
test.write('\nprediction:\n')
test.write(' '.join(generated_text))
test.close()