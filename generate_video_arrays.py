# -*- coding: utf-8 -*-
"""
A multiprocessing approach to generating and saving optical flow and rgb arrays
for each video.
The videos are saved in arrays, with videos of the same length saved together.
If the number of videos multiplied by the length of video exceeds a value which
would proclude the videos from fitting on the GPU, a new array is started
This file also saves an info dict which gives information about the saved arrays
but is not accurate. Need to run generate_dict after to fix this.
"""
from processing_utils import get_flow_rgb, normalise_rgb, normalise_flow, get_len
import numpy as np
import cv2
import skvideo.io
from pathlib import Path
import pickle
import os
from multiprocessing import Pool, Manager
from fix_batches import fix_batches


def work(inputs):
    activities = ['hand-waving','sitting-down', 'walking']
    done_list = inputs['done_list']
    doing_list = inputs['doing_list']
    info_dict = inputs['info_dict']
    doing_lock = inputs['doing_lock']
    lock = inputs['lock']
    frames_lock = inputs['frames_lock']
    max_size = 1024
    image_size = 224
    video_path = inputs['video_path']
    save_path = inputs['save_path']
    flow_path = save_path / 'flow'
    rgb_path = save_path / 'rgb'
    label_path = save_path / 'label'
    info_path = save_path / 'info_file.p'

    for activity in activities:
        activity_path = video_path / activity

        for file in os.listdir(activity_path):

            file_path = activity_path / file
                                
            doing_lock.acquire()
            if (activity + file) in done_list or (activity + file) in doing_list:
                doing_lock.release()
                continue
            doing_list.append(activity + file)
            doing_lock.release()
            # If we haven't processed the file yet, get the flow
            print('Generating flow for: \n' + str(file_path))
            flow, rgb = get_flow_rgb(str(file_path))
            
            n_frames = flow.shape[0]

            p_name = str(n_frames) + '.p'
            
            rgb_file_path = rgb_path / p_name
            flow_file_path = flow_path / p_name
            label_file_path = label_path / p_name
            
            
            flow = normalise_flow(flow)
            rgb = normalise_rgb(rgb)       
            print('Flow generated')
            frames_lock.acquire()
            if not n_frames in [f['n_frames'] for f in info_dict['files']]:
                files = info_dict['files']
                files.append({'n_frames' : n_frames, 
                                           'batch_size' : 0,
                                           'save_file' : save_path / 'features' / p_name,
                                           'sample_paths' : {'rgb' : str(rgb_file_path),
                                                             'flow' : str(flow_file_path),
                                                             'label' : str(label_file_path)}})
                info_dict.update({'files' : files})
                lock.acquire()
                frames_lock.release()
                rgb_array = np.zeros([0,n_frames, image_size, image_size, 3])
                flow_array = np.zeros([0,n_frames, image_size, image_size, 2])
                label_array = []
                
            else:
                lock.acquire()
                frames_lock.release()
                rgb_array = pickle.load(open(rgb_file_path, 'rb'))
                flow_array = pickle.load(open(flow_file_path, 'rb'))
                label_array = pickle.load(open(label_file_path, 'rb'))
            rgb_array = np.append(rgb_array, rgb[None,:,:,:,:], axis = 0)
            flow_array = np.append(flow_array, flow[None,:,:,:,:], axis = 0)
            label_array.append(activity)

            dict_index = [f['n_frames'] for f in info_dict['files']].index(n_frames)
            size = len(label_array)
            info_dict['files'][dict_index].update({'batch_size' : size})
            if size * n_frames >= max_size:
                i = 0
                while(1):
                    new_name = str(n_frames) + '_' + str(i) + '.p'
                    if not os.path.exists(label_path / new_name):
                        break
                    i += 1
                pickle.dump(rgb_array, open(rgb_path / new_name, 'wb'))
                pickle.dump(flow_array, open(flow_path / new_name, 'wb'))
                pickle.dump(label_array, open(label_path / new_name, 'wb'))
                
                rgb_array = np.zeros([0,n_frames, image_size, image_size, 3])
                flow_array = np.zeros([0,n_frames, image_size, image_size, 2])
                label_array = []
            pickle.dump(rgb_array, open(rgb_file_path, 'wb'))
            pickle.dump(flow_array, open(flow_file_path, 'wb'))
            pickle.dump(label_array, open(label_file_path, 'wb'))
            pickle.dump(dict(info_dict), open(info_path, 'wb'))
            done_list.append(activity + file)
            pickle.dump(list(done_list), open(video_path/'done_list.p','wb'))

            lock.release()
            doing_list.pop(doing_list.index(activity + file))
            

if __name__ == '__main__':
    n_workers = 8

    
    base_dir = Path('../test_videos')
    
    video_path = base_dir / 'videos'
    save_path_rel =  base_dir / 'video_arrays'
    save_path = Path(os.path.abspath(save_path_rel))
    flow_path = save_path / 'flow'
    rgb_path = save_path / 'rgb'
    label_path = save_path / 'label'
    info_path = save_path / 'info_file.p'
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(flow_path)
        os.mkdir(rgb_path)
        os.mkdir(label_path)
        
    if not os.path.exists(info_path):
        info_dict = {'saved_size' : 224,
                     'cropped_size' : 224,
                     'files' : []}
    else:
        info_dict = pickle.load(open(info_path,'rb'))
        
        
    if not os.path.exists(video_path/'done_list.p'):
        done_list = []
    else:
        done_list = pickle.load(open(video_path/'done_list.p', 'rb'))
    
    doing_list = []
    m = Manager()
    lock = m.Lock()
    frames_lock = m.Lock()
    doing_lock = m.Lock()
    done_list = m.list(done_list)
    doing_list = m.list(doing_list)
    info_dict = m.dict(info_dict)
    inputs = {'done_list' : done_list,
              'doing_list' : doing_list,
              'info_dict' : info_dict,
              'lock' : lock,
              'frames_lock' : frames_lock,
              'doing_lock' : doing_lock,
              'video_path' : video_path,
              'save_path' : save_path}
    pool = Pool(n_workers)

    pool.map(work, [inputs] * n_workers)
    
    fix_batches(base_dir)