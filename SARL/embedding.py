# from torchvision.models import resnet18
from SARL.resnet import resnet18
from matplotlib.image import imsave, imread
import torch
import numpy as np

model = resnet18(pretrained=True).float()

def get_embedding(image): #Used for converting an image into an embedding
    return model(image)


#Used for checking differences between two images
def modelPrinting(model):
    print(model)

def load_data(_list_of_files):
    return [torch.from_numpy(np.swapaxes(np.expand_dims(imread(f), axis=0), 1, 3)).float() for f in _list_of_files]

images = ["file-0.jpg", "file-1.jpg"]
_loaded_images = load_data(images)

for im in _loaded_images:
    output = model(im)
    print(output[0][:10])

# modelPrinting(model)


