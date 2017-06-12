from keras.models import load_model
from p3_models import nVidiaNet
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras import backend as tf
from p3_callbacks import PlotCallback
import numpy as np
import math
import sklearn.model_selection as sk
from p3_helperfunctions import generator
import os

# Set environment variable to suppress TF messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# This function defines and trains the nVidiaNet network
# using all the data in one shot
def train_model_keras(X_tr, Y_tr, X_va, Y_va, train_params):
    rate = train_params['LRN_RATE']['base']
    batch_size = train_params['BAT_SIZE']
    epochs = train_params['NUM_EPOC']

    model = nVidiaNet(X_tr,train_params['KEP_PROB'])
    adams_opt = optimizers.Adam(lr = rate)
    sgd_opt = optimizers.SGD(lr=rate, momentum=0.9, decay=0.1)
    model.compile(loss='mean_absolute_error',\
                  optimizer=adams_opt)
    
    if (train_params['RES_TRAN'] == True):
        print("Restoring model {}...".format(train_params['RES_FILE']))
        print("")
        model = load_model(train_params['RES_FILE'])

    
    save_model = ModelCheckpoint(filepath="ckpt_{epoch:01d}_{val_loss:.4f}.hdf5",\
                                 period=1)
    print()
    model.summary()
    print()
    print("Training...")
    print("")
    plotCallBack = PlotCallback()
    history = model.fit(x=X_tr,\
                        y=Y_tr,\
                        batch_size=batch_size,\
                        epochs=epochs,\
                        verbose=1,\
                        validation_data=(X_va,Y_va),\
                        shuffle=True,\
                        callbacks=[save_model]) #plotCallBack])
    model.save(train_params['SAV_FILE'])
    return model

# This function defines and trains the nVidiaNet network
# using a generator of certain batchsize thus requiring less memory
def train_model_keras_gen(samples, train_params):
    # Get training parameters
    rate = train_params['LRN_RATE']['base']
    batch_size = train_params['BAT_SIZE']
    epochs = train_params['NUM_EPOC']

    # Create neural network and compile model
    model = nVidiaNet(train_params['KEP_PROB'])
    adams_opt = optimizers.Adam(lr = rate)
    model.compile(loss='mean_absolute_error',\
                  optimizer=adams_opt)
    
    # Resume training if specified
    if (train_params['RES_TRAN'] == True):
        print("Restoring model {}...".format(train_params['RES_FILE']))
        print("")
        model = load_model(train_params['RES_FILE'])

    # Callback for saving model after each checkpoint
    save_model = ModelCheckpoint(filepath="ckpt_{epoch:01d}_{val_loss:.4f}.hdf5",\
                                 period=1)
    
    # Print model summary
    print()
    model.summary()
    print()
    
    # Callback for visualizing training and validation loss
    # This callback is slow, it is currently disabled
    plotCallBack = PlotCallback()

    # Split train into train and validation sets
    train_samples, validation_samples = sk.train_test_split(samples, test_size=0.2,\
                                                         random_state=42)
    print("Training on {0}, validating on {1} samples...".format(len(train_samples),len(validation_samples)))
    print("")

    # For every image, 2 more are added (random shadows + random brightness)
    # Therefore, to maintain batchsize ~ 128, specify a batchsize of 129/3=43 
    # to the generator so it returns 129 images
    use_batch_size = math.floor(batch_size/3)
    train_generator = generator(train_samples, batch_size=use_batch_size)
    validation_generator = generator(validation_samples, batch_size=use_batch_size)
    n_samples = int(math.ceil(len(train_samples)/use_batch_size))   
    v_steps = int(math.ceil(len(validation_samples)/use_batch_size))

    # Train the model
    history = model.fit_generator(generator=train_generator,\
                                  steps_per_epoch=n_samples,\
                                  validation_data=validation_generator,\
                                  validation_steps=v_steps,\
                                  epochs=epochs,\
                                  verbose=1,\
                                  callbacks=[save_model]) #plotCallBack])
    
    # Save model at the end of the training
    model.save(train_params['SAV_FILE'])
    return model