# -*- coding: utf-8 -*-
"""
This model takes the output features from the I3C model and then applies a FC
network to map them to the correct output size
"""

import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.callbacks.callbacks import ModelCheckpoint
from keras.models import Model

class FC_Model:
    def __init__(self, params):
        video_features = Input(shape = (1024,))
        _layers = [video_features]
        _layers.append(Dropout(params['input_dropout'])(_layers[-1]))
        
        for layer in params['layers']:
            _layers.append(Dense(layer, activation = 'relu')(_layers[-1]))
            _layers.append(Dropout(params['dense_dropout'])(_layers[-1]))

        output = Dense(params['n_classes'], activation = 'softmax')(_layers[-1])
        
        self.model = Model(inputs = video_features, outputs = output)
        self.model.compile(Adam(learning_rate = params['lr']), loss = sparse_categorical_crossentropy,metrics = ['accuracy'])
        self.model.summary()
        self.callback = ModelCheckpoint(params['savefile'], monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=1)
        self.val_split = params['val_split']
        self.batch_size = params['batch_size']
        
        
    def load_weights(self, savefile):
        self.model.load_weights(savefile)
        
    def learn(self, train_data, epochs):
        self.model.fit(train_data[0], train_data[1], epochs = epochs, batch_size = self.batch_size, callbacks = [self.callback], validation_split = self.val_split, shuffle = True)

    def predict(self, video_features):
        return self.model.predict(video_features)
        