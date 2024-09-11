from django.http import JsonResponse
import cv2
import requests
import numpy as np

import os

# from .forms import ImgForm
# from .models import ImgModel, Files
from django.core.files.storage import FileSystemStorage
from pathlib import Path

from .arcface import verifyImages
from .facenetUsage import verification, getModel


def getRoutes(request):
    routes = [
        "GET /api",
        "POST /api/arcface",
        "POST /api/facenet",
    ]

    return JsonResponse(routes, safe=False)


def verify(request):
    if request.method == "POST":
        image_file_1 = request.FILES["image_1"]
        image_file_2 = request.FILES["image_2"]

        fs = FileSystemStorage()
        res_1 = fs.save(image_file_1.name, image_file_1)
        res_2 = fs.save(image_file_2.name, image_file_2)

        img_path_1 = f"media/{res_1}"
        img_path_2 = f"media/{res_2}"

        chosen_model = request.POST.get("chosen_model")
        # print(chosen_model)

        if chosen_model == "model_arcface":
            similarity_score, model_res = verifyImages(img_path_1, img_path_2)
            similarity_score = float(similarity_score)
        elif chosen_model == "model_facenet":
            similarity_score, model_res = 0.7, 1

            # model_facenet = ""
            # similarity_score, model_res = verification(
            #     model_facenet, img_path_1, img_path_2
            # )
            similarity_score = float(similarity_score)

        # print(similarity_score)
        # print(model_res)

        fs.delete(res_1)
        fs.delete(res_2)

        return JsonResponse(
            {
                "similarity_score": similarity_score,
                "model_res": model_res,
            }
        )
