# importing the required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as ttf

import os
import os.path as osp
import json
import argparse

from tqdm import tqdm
from PIL import Image
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Block class for ConvNext 
class Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        '''
        in_channels: number of input channels
        '''
        #depthwise convolution
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        #point wise convolution
        self.pwconv1 = nn.Conv2d(in_channels, 4 * in_channels, kernel_size=1, stride=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * in_channels, in_channels, kernel_size=1, stride=1)


    def forward(self, x):
        '''
        x: input tensor
        return: output tensor
        '''
        input = x
        out = self.dwconv(x)
        out = self.bn(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = input + out
        return out

#ConvNext class
class ConvNext(nn.Module):
    def __init__(self, in_channels, num_classes = 7000, depths = [3, 3, 9, 3], dims = [96, 192, 384, 758]):
        super().__init__()
        '''
        in_channels: number of input channels
        num_classes: number of classes
        depths: number of blocks in each down sample layer
        dims: number of channels in each down sample layer
        '''
        self.down_sample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
                                nn.BatchNorm2d(dims[0])
                                )
        self.down_sample_layers.append(stem)
        #3 down sample layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2))
            self.down_sample_layers.append(downsample_layer)
            
        self.block_layers = nn.ModuleList()
        #4 stages of blocks
        for i in range(4):
            blocks = nn.ModuleList()
            # blocks = []
            for j in range(depths[i]):
                block = Block(dims[i])
                blocks.append(block)
            self.block_layers.append(nn.Sequential(*blocks))
        #1 final layer classification layer
        self.norm = nn.BatchNorm2d(dims[-1])
        self.classifier = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        '''
        x: input tensor
        return: output tensor
        '''
        #4 down sample layers
        for i in range(4):
            x = self.down_sample_layers[i](x)
            x = self.block_layers[i](x) 
        #finall layer
        x = self.norm(x)
        x = F.adaptive_avg_pool2d(x, 1)
        feats = x.view(x.size(0), -1)
        x = self.classifier(feats)
        return x


# Dataset class for test data
class ClassificationTestSet(Dataset):

    def __init__(self, data_dir, transforms):
        '''
        data_dir: path to the folder containing the test images
        transforms: transforms to be applied on the images
        '''
        self.data_dir = data_dir
        self.transforms = transforms

        # This one-liner basically generates a sorted list of full paths to each image in data_dir
        self.img_paths = list(map(lambda fname: osp.join(self.data_dir, fname), sorted(os.listdir(self.data_dir))))

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        return self.transforms(Image.open(self.img_paths[idx])), self.img_paths[idx].split('/')[-1]


def evaluate(model, test_loader):
    '''
    Evaluate the model on the test set
    params: model - the model to evaluate
            test_loader - the test set dataloader
    returns: preds - a list of the predicted labels for each image in the test set
    '''
    model.eval()
    batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, position=0, leave=False, desc='Test')
    res = []
    for x, file_name in test_loader:

        x = x.to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model(x)
            y_hat = torch.argmax(outputs, axis=1)
            res.extend(list(zip(file_name, y_hat.cpu().numpy())))
        
        batch_bar.update()
    batch_bar.close()
    return res



if __name__ == '__main__':

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default='mobilenet.pt',
        help="path to trained model")
    ap.add_argument("-n", "--model_name", default='mobilenet',
        help="name of the model")
    ap.add_argument("-d", "--dataset", default='test/',
        help="path to tesr folder")
    ap.add_argument("-i", "--idx_to_class", default='idx_to_class.json',
        help="path to idnex to class json file")
    args = ap.parse_args()

    # Validation transforms
    val_transforms = [
                    ttf.Resize((500, 600)),
                    ttf.CenterCrop(300),
                    ttf.ToTensor()]
 
    # Load the index to class mapping
    with open(args.idx_to_class, 'r') as f:
        idx_to_class = json.load(f)
 
    # Test directory
    TEST_DIR = args.dataset
 
    # Load the test set
    batch_size = 128
    test_dataset = ClassificationTestSet(TEST_DIR, ttf.Compose(val_transforms))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         drop_last=False, num_workers=4, pin_memory=True)

    # Load the model
    if args.model_name == 'convnext':
        model = ConvNext(3, num_classes = 9, depths = [3, 3, 9, 3], dims = [96, 192, 384, 758])
        model.to(device)
        model.load_state_dict(torch.load(args.model, map_location=torch.device(device=device)).state_dict())
    elif args.model_name == 'mobilenet':
        model = torchvision.models.mobilenet_v3_small()
        model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(576, 9))
        model.to(device)
        model.load_state_dict(torch.load(args.model, map_location=torch.device(device=device)).state_dict())

    # Evaluate the model
    res = evaluate(model, test_loader)
    # map the labels to the classes
    res = list(map(lambda x: (x[0], idx_to_class[str(x[1])]), res))

    # Create a dataframe with the results
    df = pd.DataFrame(res, columns=['file_name', 'predicted_class'])
    # Print the dataframe line by line
    for i in range(len(df)):
        print('File name: {}, Predicted class: {}'.format(df.iloc[i]['file_name'], df.iloc[i]['predicted_class']))

    print('-'*100)
    #print the number of images in each class
    print('Predicted class:', 'Number of images')
    print(df['predicted_class'].value_counts())