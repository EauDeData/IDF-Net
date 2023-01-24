import torch
from vit_pytorch import ViT


class VisualTransformer(torch.nn.Module):
    def __init__(self, image_size, patch_size = 32, embedding_size = 128, depth = 1, heads = 1, dropout = 0.1) -> None:
        super(VisualTransformer, self).__init__()
        self.extractor = ViT(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = 1,
            dim = embedding_size,
            depth = depth,
            heads = heads,
            mlp_dim = 1,
            dropout = dropout,
            emb_dropout = dropout
        )
        self.extractor.mlp_head = list(self.extractor.mlp_head.children())[-2]

    
    def forward(self, batch):
        return self.extractor(batch)

class Resnet50(torch.nn.Module):
    def __init__(self, embedding_size = 128):
        super(Resnet50, self).__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=False)
        self.resnet50.fc = torch.nn.Linear(2048, embedding_size)
    def forward(self, batch):
        return self.resnet50(batch)
        

class VisualConvTransformer(torch.nn.Module):
    def __init__(self) -> None:
        pass
    def forward(self, batch):
        pass