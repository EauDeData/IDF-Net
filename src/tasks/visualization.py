import torch
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM

# https://open.spotify.com/track/0tZYFh6kyraGwBs5VSR3HL?si=5537544b586a46a8


# TODO: Test it :-3

class SimilarityToConceptTarget:

    # Source: https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/Pixel%20Attribution%20for%20embeddings.ipynb

    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(model_output, self.features)
    

class SelfSimilarityTarget:

    # Does this make any sense? 

    def __init__(self, augm = True, augm_magnitude = 1e-3):
        self.augm = augm_magnitude if augm else 0
    def __call__(self, model_output):

        random_augm = self.augm * (torch.rand_like(model_output) * 2 - 1)
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(model_output, model_output + random_augm)

class GradCamScanner:
    def __init__(self, model, target, layers, device = 'cuda') -> None:
        self.model = model
        self.target = target
        self.target_layers = layers
        self.use_cuda = device == 'cuda'

    def scan(self, image):
        with GradCAM(model=self.model, target_layers=self.target_layers, use_cuda=False) as cam:
            return cam(input_tensor=image[None,], targets=[self.target])[0, :]


