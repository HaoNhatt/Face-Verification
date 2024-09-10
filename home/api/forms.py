from django import forms


class ImgForm(forms.Form):
    image_field = forms.ImageField()
