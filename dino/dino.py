import pytorch_lightning as pl
import requests
from PIL.Image import Image
import  dino_featurizer



def my_app():


    model = dino_featurizer.DinoFeaturizer(3,"vit_small",16)


if __name__ == "__main__":
    my_app()



