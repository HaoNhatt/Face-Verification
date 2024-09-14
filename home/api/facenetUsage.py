import torch
import PIL
from torchvision import transforms
from backbones.InceptionResnetV1_facenetpytorch import *


def getModel(nClasses, path):
    """
    Create a InceptionResnetV1 model with nClasses output classes.
    Load saved weight from path.
    """
    model = InceptionResnetV1(num_classes=nClasses)
    model.load_state_dict(torch.load(path))
    return model


model = getModel(nClasses = 2154, path = "../../models/facenet_model.pth")
model.eval()

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


def loadImage2Tensor(path):
    """
    Load image from path, transform and return its tensor form.
    """
    image = PIL.Image.open(path)
    trans = transforms.Compose(
        [
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            fixed_image_standardization,
        ]
    )
    return trans(image).unsqueeze(0)


def computeEmbedding(img):
    """
    Return the L2 normalised embedding of the input image.
    """
    flattenOutput = torch.flatten(model(img).detach().cpu())
    return torch.nn.functional.normalize(flattenOutput, p=2.0, dim=0)


def cosineSimilarity(emb1, emb2):
    sim = torch.dot(emb1, emb2)
    return sim


def verification(path1, path2, threshold=0.6803):
    """
    Return the cosine similarity between two embeddings
    and whether they are of the same person.
    """
    img1 = loadImage2Tensor(path1)
    img2 = loadImage2Tensor(path2)
    emb1 = computeEmbedding(img1)
    emb2 = computeEmbedding(img2)
    sim = cosineSimilarity(emb1, emb2)
    return sim, sim >= threshold

