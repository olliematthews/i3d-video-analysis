# -*- coding: utf-8 -*-
"""
Used to generate a arrays of training data from the extracted features and 
labels
"""

import numpy as np
import pickle
from pathlib import Path

def join_arrays(path):
    base_dir = Path(path)
    info_dict = pickle.load(open(base_dir / 'info_file.p','rb'))
    X_rgb = np.zeros([0,1024])
    X_flow = np.zeros([0,1024])
    y = []
    for file in info_dict['files']:
        savefile = file['save_file']
        rgb_flow = pickle.load(open(savefile, 'rb'))
        n_reps = rgb_flow[0].shape[1]
        X_rgb = np.append(X_rgb, rgb_flow[0].reshape([-1,1024]), axis = 0)
        X_flow = np.append(X_flow, rgb_flow[1].reshape([-1,1024]), axis = 0)
        labelfile = file['sample_paths']['label']
        y.extend(list(np.repeat(pickle.load(open(labelfile, 'rb')), n_reps)))
        
    pickle.dump(X_rgb, open('rgb_train.p', 'wb'))
    pickle.dump(X_flow, open('flow_train.p', 'wb'))
    pickle.dump(y, open('y_train.p', 'wb'))
    
if __name__ == '__main__':
    join_arrays('../video_arrays')