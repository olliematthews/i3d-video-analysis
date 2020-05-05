# -*- coding: utf-8 -*-
"""
The function in here goes through the resultant arrays from generate_video_arrays
and generates a dictionary to describe the resultant arrays for use in 
the feature extractor.
"""
import pickle
import numpy as np
import os
from pathlib import Path

def fix_batches(base_dir):
    base_dir = Path(base_dir)
    video_dir = Path( base_dir / 'video_arrays')
    info_dict = pickle.load(open(video_dir / 'info_file.p','rb'))
    files = []
    
    ns = [f['n_frames'] for f in info_dict['files']]
    label_path = Path(os.path.abspath(video_dir / 'label'))
    rgb_path = Path(os.path.abspath(video_dir / 'rgb'))
    flow_path = Path(os.path.abspath(video_dir / 'flow'))
    feature_path = Path(os.path.abspath(video_dir / 'features'))
    
    if not os.path.exists(feature_path):
        os.mkdir(feature_path)
    
    for file in os.listdir(label_path):
        if file[-4] == '_':
            n_frames = int(file.split('_')[0])
        else:
            n_frames = int(file.split('.')[0])
        sample_paths = {'rgb' : rgb_path /  file,
                        'flow' : flow_path / file,
                        'label' : label_path / file}
        save_file = feature_path / file
        batch_size = len(pickle.load(open(label_path/file, 'rb')))
        if batch_size == 0:
            os.remove(rgb_path /  file)
            os.remove(flow_path /  file)
            os.remove(label_path /  file)
            continue
        files.append({
                'n_frames' : n_frames,
                'sample_paths' : sample_paths,
                'save_file' : save_file,
                'batch_size' : batch_size})
    info_dict.update({'files' : files})
    pickle.dump(info_dict, open(video_dir / 'info_file.p','wb'))