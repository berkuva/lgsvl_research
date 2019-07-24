import torch
from torch.utils import data
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


FILEPATH = "/home/derek/Desktop/simulator/Api/examples/NHTSA-sample-tests/Encroaching-Oncoming-Vehicles/"

class Dataset(data.Dataset):
	def __init__(self, list_IDs, labels):
		self.list_IDs = list_IDs
		self.labels = labels

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		X = self.list_IDs[index]
		y = self.labels[str(index)+"main-camera.jpg"]
		y = torch.tensor([y])
		return X, y

##### IMAGE PROCESSING #####
pil2tensor = transforms.ToTensor()
tensor2pil = transforms.ToPILImage()

def read_image(image_path):
	pil_image = Image.open(image_path)
	rgb_image = pil2tensor(pil_image)
	return rgb_image

def show_image(image_tensor):
	plt.figure()
	plt.imshow(image_tensor.numpy().transpose(1,2,0))
	plt.show()

partition = {}
labels = {}

scenes = [f for f in listdir(FILEPATH+"training/") if isfile(join(FILEPATH+"training/", f))]

for i, s in enumerate(scenes):
	training_rgb = read_image(FILEPATH+"training/"+str(i)+"main-camera.jpg")
	waypoint_rgb = read_image(FILEPATH+"waypoints/"+str(i)+"main-camera.jpg")
	if len(partition) > 0:
		partition["training"].append(training_rgb)
		partition["waypoints"].append(waypoint_rgb)
	else:
		partition["training"] = [training_rgb]
		partition["waypoints"] = [waypoint_rgb]
	labels[str(i)+"main-camera.jpg"] = i

# Parameters
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 6}

show_image(partition["waypoints"][-1])
# Generators
scene_set = Dataset(partition['training'], labels)
scene_generator = data.DataLoader(scene_set, **params)

waypoint_set = Dataset(partition['waypoints'], labels)
waypoint_generator = data.DataLoader(waypoint_set, **params)

for scene, label in scene_generator:
	scene, label = scene.to(device), label.to(device)
	# print(scene.shape)
	# print(label.shape)