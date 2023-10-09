from torchvision.models import resnet50, ResNet50_Weights
#from resnet import resnet18
from matplotlib.image import imsave, imread
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
#print(torch.cuda.is_available())
model = resnet50(weights=ResNet50_Weights.DEFAULT).float().to(device)#pretrained=True).float().to(device)#, non_blocking=True)


def get_embedding(image): #Used for converting an image into an embedding
    return model(torch.from_numpy(image).float().to(device)).to(device)#, non_blocking=True))


#Used for checking differences between two images
def modelPrinting(model):
    print(model)


def load_data(_list_of_files):
    return [torch.from_numpy(np.swapaxes(np.expand_dims(imread(f), axis=0), 1, 3)).float() for f in _list_of_files]

# images = ["file-0.jpg", "file-1.jpg"]
# _loaded_images = load_data(images)
#
# for im in _loaded_images:
#     output = model(im)
#     print(output[0][:10])

# modelPrinting(model)


