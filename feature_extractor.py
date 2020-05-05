# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Closely based on the evaluate_sample code at:
    
https://github.com/deepmind/kinetics-i3d

Copies the model which is pretrained on ImageNet and Kinetic and uses it to 
output features for our videos
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path
import sys

rel_path = '../kinetics-i3d'
sys.path.insert(1, rel_path)
import i3d



_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_CHECKPOINT_PATHS = {k : rel_path + '/' + v for k, v in _CHECKPOINT_PATHS.items()}
    

_END_POINT = 'Mixed_5c'


def main():
  tf.logging.set_verbosity(tf.logging.INFO)
  eval_type = 'joint'

  imagenet_pretrained = 'imagenet_pretrained'

  NUM_CLASSES = 400
  if eval_type == 'rgb600':
    NUM_CLASSES = 600

  if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
    raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')

  if eval_type in ['rgb', 'rgb600', 'joint']:
    # RGB input has 3 channels.
    rgb_input = tf.placeholder(
        tf.float32,
        shape=(_BATCH_SIZE, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))


    with tf.variable_scope('RGB', reuse = _REUSE):
      rgb_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint=_END_POINT)
      rgb_features, _ = rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)
#      rgb_features = tf.nn.avg_pool3d(rgb_features, ksize=[1, 2, 7, 7, 1],
#                             strides=[1, 1, 1, 1, 1], padding=snt.VALID)
#      rgb_features = tf.squeeze(rgb_features, [2, 3], name='SpatialSqueeze')

    rgb_variable_map = {}
    for variable in tf.global_variables():

      if variable.name.split('/')[0] == 'RGB':
        if eval_type == 'rgb600':
          rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
        else:
          rgb_variable_map[variable.name.replace(':0', '')] = variable

    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

  if eval_type in ['flow', 'joint']:
    # Flow input has only 2 channels.
    flow_input = tf.placeholder(
        tf.float32,
        shape=(_BATCH_SIZE, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
    with tf.variable_scope('Flow', reuse = _REUSE):
      flow_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint=_END_POINT)
      flow_features, _ = flow_model(
          flow_input, is_training=False, dropout_keep_prob=1.0)
#      flow_features = tf.nn.avg_pool3d(flow_features, ksize=[1, 2, 7, 7, 1],
#                             strides=[1, 1, 1, 1, 1], padding=snt.VALID)
#      flow_features = tf.squeeze(flow_features, [2, 3], name='SpatialSqueeze')

    flow_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'Flow':
        flow_variable_map[variable.name.replace(':0', '')] = variable
    flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

  with tf.Session() as sess:
    feed_dict = {}
    if eval_type in ['rgb', 'rgb600', 'joint']:
      if imagenet_pretrained:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
      else:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
      tf.logging.info('RGB checkpoint restored')
      rgb_sample = pickle.load(open(_SAMPLE_PATHS['rgb'],'rb'))
      # Take the random crops
      rgb_list = []
      for video, i, j in zip(rgb_sample, _CROP_STARTS[:,0], _CROP_STARTS[:,1]):
          rgb_list.append(video[:,i : i + _IMAGE_SIZE, j : j + _IMAGE_SIZE, :])
      rgb_sample = np.array(rgb_list)
      tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
      feed_dict[rgb_input] = rgb_sample

    if eval_type in ['flow', 'joint']:
      if imagenet_pretrained:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
      else:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
      tf.logging.info('Flow checkpoint restored')
      flow_sample = pickle.load(open(_SAMPLE_PATHS['flow'],'rb'))
      # Take the random crops
      flow_list = []
      for video, i, j in zip(flow_sample, _CROP_STARTS[:,0], _CROP_STARTS[:,1]):
          flow_list.append(video[:,i : i + _IMAGE_SIZE, j : j + _IMAGE_SIZE, :])
      flow_sample = np.array(flow_list)
      tf.logging.info('Flow data loaded, shape=%s', str(flow_sample.shape))
      feed_dict[flow_input] = flow_sample

    out_rgb, out_flow = sess.run(
        [rgb_features, flow_features],
        feed_dict=feed_dict)
    
    # Apply the mean pool before saving the arrays
    out_rgb = np.mean(out_rgb, axis = (2,3))
    out_flow = np.mean(out_flow, axis = (2,3))
    pickle.dump([out_rgb, out_flow], open(_SAVE_FILE,'wb'))

if __name__ == '__main__':
    np.random.seed(0)
    base_dir = Path('../test-videos')
    arrays_dir = base_dir / 'video_arrays'
    info_path = Path(arrays_dir / 'info_file.p')
    info_dict = pickle.load(open(info_path, 'rb'))
    
    _IMAGE_SIZE = info_dict['cropped_size']
    _REUSE = False
    for file_info in info_dict['files']:
        _SAMPLE_VIDEO_FRAMES = file_info['n_frames']
        _SAMPLE_PATHS = file_info['sample_paths']
        _BATCH_SIZE = file_info['batch_size']
        _SAVE_FILE = file_info['save_file']
        if not _IMAGE_SIZE == info_dict['saved_size']:
            _CROP_STARTS = np.random.randint(0, info_dict['saved_size'] - _IMAGE_SIZE, (_BATCH_SIZE, 2))
        else: 
            _CROP_STARTS = np.zeros([_BATCH_SIZE, 2]).astype(int)
        main()
        _REUSE = True