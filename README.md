The code here will generate and analyse features extracted using the I3C model for videos. To use the code run:

For training and test data:

generate_video_arrays - (ensure base_dir points to a directory containing the videos, with each category in its own directory.

feature_extractor - to run this, you need to download the directory containing the I3C and make sure the correct address is put in. The directory can be found at: https://github.com/deepmind/kinetics-i3d


For training data:

training-utils - (needs to be run on the training data)

train_model


For test data:

evaluate_model
