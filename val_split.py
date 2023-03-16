import os
import argparse
import numpy as np


# function to move 20% of the frames to a validation folder
def val_split(path):
    '''
    :param path: path to the folder containing the frames
    :return: None
    '''
    # find the folders in the path
    folders = [f for f in os.listdir(path) if len(f) == 5]
    folders.sort()
    # loop through the folders
    for folder in folders:
        if not os.path.exists('val/' + folder):
            os.makedirs('val/' + folder)
        # find the files in the folder
        files = [f for f in os.listdir(path + folder) if f.endswith('.png')]
        files.sort()
        # randomly select 20% of the files
        val_files = np.random.choice(files, int(len(files) * 0.2), replace=False)
        # move the files to the validation folder
        for file in val_files:
            os.rename(path + folder + '/' + file, 'val/' + folder + '/' + file)

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", type=str, default="train/", help="path to the folder containing the frames")
    args = ap.parse_args()
    # call the function
    val_split(args.path)
