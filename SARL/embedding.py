from torchvision.models import densenet201, googlenet
from resnet import resnet18, resnet50, resnet152
from matplotlib.image import imsave, imread
import torch
import numpy as np
device = torch.device("cuda")

#model = torch.hub.load("facebookresearch/swag", model="regnety_32gf_in1k").float().to(device, non_blocking=True)
#print(device)
#print(torch.cuda.is_available())
model = resnet50(pretrained=True).float().to(device)#resnet18(pretrained=True).float().to(device)#pretrained=True).float().to(device)#, non_blocking=True)
#print(model)

def get_embedding(image): #Used for converting an image into an embedding
    global model, device
    return model(torch.from_numpy(image).float().to(device)).to(device, non_blocking=True)


#Used for checking differences between two images
def modelPrinting(model):
    print(model)


def load_data(_list_of_files):
    return [torch.from_numpy(np.swapaxes(np.expand_dims(imread(f), axis=0), 1, 3)).float() for f in _list_of_files]

def save_embedding_csv(file_name, data):#895 warplane, 864 tow truck, 817 sportscar, 751 racecar, 653 milkcan
    filey = open((file_name[:-3]+"csv"), "w")
    print(data.shape)
    for d in data[0]:
        filey.write(str(d) + "\n")

    filey.close()
#images = ["file-0.jpg", "file-1.jpg"]
#_loaded_images = load_data(images)

#for im, fi in zip(_loaded_images, images):
#    output = model(im)
#    save_embedding_csv(fi, output.detach().cpu().numpy())
