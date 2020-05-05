# -*- coding: utf-8 -*-
"""
Test the performance of the model which was trained in train_model.
"""
from model import FC_Model
import numpy as np
import pickle

rgb_params, flow_params = pickle.load(open('model.p','rb'))
classes = ['walking', 'sitting-down', 'hand-waving']

# Initialise the models
rgb_model = FC_Model(rgb_params)
flow_model = FC_Model(flow_params)

# Load the weights
rgb_model.load_weights(rgb_params['loadfile'])
flow_model.load_weights(flow_params['loadfile'])

info_dict = pickle.load(open('../test-videos/video_arrays/info_file.p', 'rb'))

class_counts = {'walking' : 0, 'sitting-down' : 0, 'hand-waving' : 0}
total = 0
rgb_correct = 0
flow_correct = 0
combined_correct = 0

seen = []
for file in info_dict['files']:
    feature_path = file['save_file']
    label_path = file['sample_paths']['label']
    
    features = pickle.load(open(feature_path,'rb'))
    labels = pickle.load(open(label_path, 'rb'))
    rgb_features = features[0]
    flow_features = features[1]
    
    for i in range(len(labels)):
        rgb = rgb_features[i]
        flow = flow_features[i]
        label = labels[i]
            
        rgb_probs = rgb_model.predict(rgb)
        flow_probs = flow_model.predict(flow)

        rgb_prob = np.sum(rgb_probs, axis = 0) / np.sum(rgb_probs)
        flow_prob = np.sum(flow_probs, axis = 0)/ np.sum(flow_probs)
        combined_prob = (flow_prob + rgb_prob) / 2
        
        rgb_pred = np.argmax(rgb_prob)
        flow_pred = np.argmax(flow_prob)
        combined_pred = np.argmax(combined_prob)
        
        # This section is here to deal with repititions which cropped up in the
        # test data for 'hand-waving' due to a bug.
        if np.linalg.norm(flow) in seen:
            print('Duplicate! Skipping...')
            continue
        else:
            seen_waving.append(np.linalg.norm(flow))
            
            
        class_counts[label] += 1
        total += 1
        if rgb_pred == classes.index(label):
            rgb_correct += 1
        if flow_pred == classes.index(label):
            flow_correct += 1
        if combined_pred == classes.index(label):
            combined_correct += 1
            
print(f'rgb accuracy - {rgb_correct / total}')
print(f'flow accuracy - {flow_correct / total}')
print(f'combined accuracy - {combined_correct / total}')
print(class_counts)

