import annoy
import os
import torch

class Annoyifier:

    '''
    Creates an annoy instance given a dataset (train data)
    for both visual and textual models.
    This will be used for evaluating ranks (visual) with respect a true rank (textual)
    

    WARNING: Dataset has to be loader with its tokenizer so it returns the text embedding properly.
    '''

    def __init__(self, train_set, visual_model, fvisual, ftext, distance = 'angular', ntrees = 10, visual = './dataset/visual.ann', text = './dataset/text.ann', device = 'cuda') -> None:

        self.text_tree = annoy.AnnoyIndex(ftext, distance)
        self.visual_tree = annoy.AnnoyIndex(fvisual, distance)

        text_done = os.path.exists(text)
        visual_done = os.path.exists(visual)
        if text_done: self.text_tree.load(text)
        if visual_done: self.visual_tree.load(visual)
        visual_model = visual_model.to(device)
        self.device = device

        if not (visual_done and text_done):

            print("Annoying models...")
            with torch.no_grad():
                for idx in range(len(train_set)):
                    print(idx, '\t', end = '\r')

                    data, text_data = train_set[idx]
                    if not visual_done:
                        data = torch.from_numpy(data).unsqueeze(0)
                        data = data.to(device)
                        self.visual_tree.add_item(idx, visual_model(data).cpu().squeeze().detach().numpy())
                        del data

                    if not text_done:
                        self.text_tree.add_item(idx, text_data)
                        del text_data
                
                if not text_done:
                    self.text_tree.build(ntrees)
                    self.text_tree.save(text)
                
                if not visual_done:
                    self.visual_tree.build(ntrees)
                    self.visual_tree.save(visual)
        self.model = visual_model
    
    def retrieve_vector(self, v, k = 10, mode: str = 'text'):

        if mode == 'text': out = self.text_tree.get_nns_by_vector(v, k, include_distances = True)
        elif mode == 'visual': out = self.visual_tree.get_nns_by_vector(v, k, include_distances = True)
        return out

    def retrieve_image(self, img, k = 10):
        with torch.no_grad():
            img = torch.from_numpy(img)
            v = self.model(img.to(self.device).unsqueeze(0)).squeeze().cpu().detach().numpy()

        return self.retrieve_vector(v, k, mode = 'visual')