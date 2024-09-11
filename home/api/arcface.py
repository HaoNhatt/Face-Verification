from .backbones import get_model
import cv2
import torch
import numpy as np
from numpy.linalg import norm


def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))


def extract_embedding(img_path, net):
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)  # Normalize the image

    feat = net(img).detach().numpy().flatten()

    return feat


def compare_uploaded_images(net, img_path_1, img_path_2, threshold=0.5):
    img1_path = img_path_1
    img2_path = img_path_2

    embedding1 = extract_embedding(img1_path, net)
    embedding2 = extract_embedding(img2_path, net)

    similarity = cosine_similarity(embedding1, embedding2)
    model_res = 0

    if similarity > threshold:
        print(
            f"The two images are likely from the SAME identity (Similarity: {similarity:.4f})."
        )
        model_res = 1
    else:
        print(
            f"The two images are likely from DIFFERENT identities (Similarity: {similarity:.4f})."
        )

    return similarity, model_res


def load_model(weight_path, model_name="r50"):
    net = get_model(model_name, fp16=False)
    net.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    net.eval()
    return net


def verifyImages(img_path_1, img_path_2):
    # print(torch.cuda.is_available())
    model_path = "models/arcface_model.pt"

    net = load_model(model_path)

    similarity_score, model_res = compare_uploaded_images(
        net, img_path_1, img_path_2, threshold=0.1076
    )
    return similarity_score, model_res
