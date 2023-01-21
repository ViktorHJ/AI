import os
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.datasets as dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

NUM_WORKERS=int(os.cpu_count() / 2)
batch_size, SHUFFLE, PINMEM = 8, True, True

# the training transforms
train_transform = transforms.Compose([
    transforms.Resize((240,320)),
    transforms.Grayscale(num_output_channels=1), #convert to grayscale
    transforms.ToTensor,
    transforms.Normalize([0.485], [0.229]) # normalize with mean and std oftransforms.Grayscale(num_output_channels=1)
])


import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import re
import os
from os import listdir
from os.path import isfile, join

def extract_numbers_from_string(string):
    return int(re.findall(r'\d+', string)[0])

image_folder = 'F:\Downloads\DATASET\TwoLocation\Training\Color\jpg'
label_folder = 'F:\Downloads\DATASET\TwoLocation\Training\Infrared\jpg'

image_names = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]
label_names = [f for f in listdir(label_folder) if isfile(join(label_folder, f))]

mapping = {}
for image_name in image_names:
    number = extract_numbers_from_string(image_name)
    label_name = 'label ({}).jpg'.format(number)
    mapping[image_name] = label_name

from torch.utils.data import Dataset

class ImageLabelDataset(Dataset):
    def __init__(self, image_folder, label_folder, mapping,transform):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.mapping = mapping
        self.transform = transform
        

    def __getitem__(self, index):
        image_name = list(self.mapping.keys())[index]
        label_name = self.mapping[image_name]

        # read the image and label using the path
        image = plt.imread(os.path.join(self.image_folder, image_name))
        label = plt.imread(os.path.join(self.label_folder, label_name))

        return image, label

    def __len__(self):
        return len(self.mapping)
    
dataset = ImageLabelDataset(image_folder, label_folder, mapping,transform=train_transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=1, num_workers=4, pin_memory=1)



if __name__ == '__main__':
    # initialize lists to store the training and validation losses
    train_losses = []

    # Create the U-Net architecture
    class UNet(nn.Module):
        def __init__(self):
            super(UNet, self).__init__()
            # Define the encoder part of the U-Net
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                #nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), FULL COLOR
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            
            # Define the decoder part of the U-Net
            self.decoder = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=4, mode='nearest'),
                nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True)
            )



        def forward(self, x):
            # pass the input through the encoder
            x = self.encoder(x)

            # pass the encoded features through the decoder
            x = self.decoder(x)

            return x

    # Create the model and move it to the device
    model = UNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function and optimizer
    # define the criterion
    criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    # define the optimizer
    #optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_steps=2000 # default needs to change prob
    # define the scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-6)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        # loop through the training data
        for i, (inputs, labels) in enumerate(dataloader):
            print("the indiex is :", i,"  rest is:  ",inputs.shape, labels.shape)
            # forward pass
            inputs = inputs.float()
            labels = labels.float()
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
            # backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

            # store the training loss
            train_losses.append(loss.item())

            print('Epoch: {} Loss: {:.4f}'.format(epoch+1, loss.item()))

    # plot the training and validation losses
    plt.plot(train_losses, label='Training Loss')
    plt.legend()
    plt.show()

    ('Training complete')