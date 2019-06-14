import cv2
import os
import argparse
from demo_FSANET_yolov2 import main
parser = argparse.ArgumentParser('Param For Head Pose Estimator')
parser.add_argument('--video',help='Your Video\'s Path To Demo',required=True)
parser.add_argument('--output',help='Your Output Video\'s Path')


def detect_video(video_path,output_path=''):
    if not os.path.exists(video_path):
        raise OSError('Video is not exists')
    else:
        if output_path != '': 
            main(video_path,output_path)
        else:
            main(video_path)

if __name__ =='__main__':
    args = parser.parse_args()
    video_path = args.video
    output_path = args.output
    detect_video(video_path,output_path)