from django.db import models


class ImgModel(models.Model):
    image_model = models.ImageField(upload_to="images/")
    # image_2 = forms.ImageField(upload_to="image")


class Files(models.Model):
    file = models.FileField(upload_to="file")
