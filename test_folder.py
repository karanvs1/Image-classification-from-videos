import os
import argparse
import shutil

# function to copy all files from val to test
def copy_to_test(path):
    '''
    :param path: path to the folder containing the frames
    :return: None
    '''
    # find the folders in the path
    folders = [f for f in os.listdir(path) if len(f) == 5]
    folders.sort()
    # loop through the folders
    for folder in folders:
        if not os.path.exists('test/'):
            os.makedirs('test/')
        # find the files in the folder
        files = [f for f in os.listdir(path + folder) if f.endswith('.png')]
        files.sort()
        # copy all files to test folder
        for file in files:
            shutil.copy(path + folder + '/' + file, 'test/' + file)

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", type=str, default="val/", help="path to the folder containing the frames")
    args = ap.parse_args()
    # call the function
    copy_to_test(args.path)