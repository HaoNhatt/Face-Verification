import torch
import PIL
from torchvision import transforms
from backbones.InceptionResnetV1_facenetpytorch import *

def getModel(nClasses, path):
    '''
    Create a InceptionResnetV1 model with nClasses output classes.
    Load saved weight from path.
    '''
    model = InceptionResnetV1(num_classes = nClasses)
    model.load_state_dict(torch.load(path)['state_dict'])
    return model

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

def loadImage2Tensor(path):
    '''
    Load image from path, return its tensor form.
    '''
    image = PIL.Image.open(path)
    trans = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    return trans(image).unsqueeze(0)


def computeEmbedding(model, img):
    '''
    Return the L2 normalised embedding of the input image.
    '''
    flattenOutput = torch.flatten(model(img).detach().cpu())
    return torch.nn.functional.normalize(flattenOutput, p = 2.0, dim = 0)

def cosineSimilarity(emb1, emb2, threshold = 0.6803):
    '''
    Return the cosine similarity between two embeddings (float [0, 1])
        and whether they are of the same person (bool).
    '''
    sim = torch.dot(emb1, emb2)
    return sim, sim > threshold