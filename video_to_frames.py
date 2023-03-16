# import libraries
import cv2
import os
import argparse


# function to extract frames
def FrameCapture(path):
    '''
    :param path: path to the video file
    :return: None
    '''
    # Path to the folder containing the videos
    # find the video files in the folder
    video_files = [f for f in os.listdir(path) if f.endswith('.mp4')]
    video_files.sort()
    # loop through the video files
    currentframe = 0
    # loop through the video files
    for video in video_files:
        # create a folder to store the frames
        folder_name = video.split('.')[0]
        #check if the folder name ends with a light or dark
        if folder_name.endswith('light'):
            folder_name = 'train/'+folder_name[:-5]
           
        if folder_name.endswith('dark'):
            folder_name = 'train/'+folder_name[:-4]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # read the video from specified path
        cam = cv2.VideoCapture(path + video)

        # loop through the frames
        while(True):
            ret, frame = cam.read()
            if ret:
                if currentframe % 2 == 0: # extract every second frame
                    name = './' + folder_name + '/frame' + str(currentframe) + '.png'
                    print ('Creating...' + name)
                    cv2.imwrite(name, frame)
                currentframe += 1
            else:
                break
        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()

if  __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", type=str, default="ML Assessment Data/", help="path to the folder containing the videos")
    args = ap.parse_args()
    # call the function
    FrameCapture(args.path)

