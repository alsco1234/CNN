import cv2
import argparse
import glob
import os
import time

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def video2image(opt):
    video_paths = glob.glob(opt.video_dir+"/*.mp4")
    print(video_paths)
    for vp in video_paths:
        video_name = vp.split("/")[-1][:-4]
        vidcap = cv2.VideoCapture(vp)
        success,image = vidcap.read()
        count = 0
        new_path = '/projects/vode/data/kinetics400_2_Images/test/abseiling'

        createDirectory(new_path+video_name)

        while success:
            cv2.imwrite(new_path+video_name+"/%04d.jpg" % count, image)     # save frame as JPEG file
            success,image = vidcap.read()
            count += 1
    
        print(video_name + " : all convert finish!!")


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, help='path to video directory')
    opt = parser.parse_args()

    video2image(opt)
    
    end = time.time()
    print(f"{end - start:.10f} sec")

    # CCD - crash : /projects/vode/data/CCD/Crash-1500
            # new_path : /projects/vode/data/CCD_images/Crash-1500/
    # CCD = normal : /projects/vode/data/CCD/Normal
            # new_path : /projects/vode/data/CCD_images/Normal/
