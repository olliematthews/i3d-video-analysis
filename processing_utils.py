'''
Code for generating the optical flow and normalising videos. Based on Koki's 
code from :
https://github.com/liwii/naked-people-tools

The normalisation is done in the same way as the original I3C paper.
'''
import numpy as np
import cv2
import skvideo.io
from pathlib import Path

def get_len(filepath:str):
    # Get the number of frames in a video
    cap = cv2.VideoCapture(filepath)
    length =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length
    
def get_flow_rgb(filepath: str):
    # Get the optical flow using OpenCV
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

    # flow.shape (2, frame_count, height, width)
    # flow[0] is dx, flow[1] is dy
    # Use TV-L1 algorithm instead, since it is used in mlb-youtube dataset
    # https://www.ipol.im/pub/art/2013/26/article.pdf

    cap = cv2.VideoCapture(filepath)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _, frame1 = cap.read()
    prvf = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    flow_video = []
    rgb_video = [frame1]
    for i in range(length - 1):
        success, frame2 = cap.read()
        if not success:
            break
        rgb_video.append(frame2)
        nextf = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        optical_flow = cv2.optflow.createOptFlow_DualTVL1()
        flow_frame = optical_flow.calc(prvf, nextf, None)
        # flow_frame = cv2.calcOpticalFlowFarneback(prvf,nextf, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_video.append(flow_frame)
        prvf = nextf
    cap.release()
    flow_video = np.array(flow_video)
    rgb_video = np.array(rgb_video)
    return flow_video, rgb_video[:-1]

def normalise_rgb(rgb):
    # For rgb we just center and normalise
    rgb = rgb.astype(np.float16)
    rgb -= 127.5
    rgb /= 127.5
    return rgb
    
def normalise_flow(flow):
    # For the flow, we truncate to +- 20, and then normalise
    flow = np.clip(flow, -20,20)
    flow /= 20
    return flow
    