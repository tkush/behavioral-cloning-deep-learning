from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.initializers import RandomNormal, Zeros, glorot_normal
from keras import regularizers

# This function defines the nVidia model
def nVidiaNet(dropout, beta=0.):
    seed = 1000
    xa_init = glorot_normal(seed)
    model = Sequential()
    model.add(Conv2D(filters=32, \
                   kernel_size=(3,3), \
                   padding='valid', \
                   activation='relu', \
                   use_bias=True, \
                   data_format='channels_last', \
                   kernel_initializer=xa_init, \
                   bias_initializer='zeros', \
                   input_shape=(40,160,3), \
                   kernel_regularizer=regularizers.l2(beta)))

    model.add(MaxPooling2D(pool_size=(2,2), \
                       strides=(2,2), \
                       padding='valid',\
                       data_format='channels_last'))

    model.add(Conv2D(filters=64, \
                   kernel_size=(3,3), \
                   padding='valid', \
                   activation='relu', \
                   use_bias=True, \
                   data_format='channels_last', \
                   kernel_initializer=xa_init, \
                   bias_initializer='zeros', \
                   kernel_regularizer=regularizers.l2(beta)))

    model.add(MaxPooling2D(pool_size=(2,2), \
                       strides=(2,2), \
                       padding='valid',\
                       data_format='channels_last'))
                       
    model.add(Conv2D(filters=128, \
                   kernel_size=(3,3), \
                   padding='valid', \
                   activation='relu', \
                   use_bias=True, \
                   data_format='channels_last', \
                   kernel_initializer=xa_init, \
                   bias_initializer='zeros', \
                   kernel_regularizer=regularizers.l2(beta)))

    model.add(MaxPooling2D(pool_size=(2,2), \
                       strides=(2,2), \
                       padding='valid',\
                       data_format='channels_last'))
    
    model.add(Flatten())
    model.add(Dense(units=500,\
                activation='relu', \
                use_bias=True, \
                kernel_initializer=xa_init, \
                bias_initializer='zeros', \
                kernel_regularizer=regularizers.l2(beta)))
    model.add(Dropout(rate=dropout,\
                      seed=100))
    model.add(Dense(units=100,\
                activation='relu', \
                use_bias=True, \
                kernel_initializer=xa_init, \
                bias_initializer='zeros', \
                kernel_regularizer=regularizers.l2(beta)))
    model.add(Dropout(rate=dropout,\
                      seed=100))
    model.add(Dense(units=20,\
                activation='relu', \
                use_bias=True, \
                kernel_initializer=xa_init, \
                bias_initializer='zeros', \
                kernel_regularizer=regularizers.l2(beta)))
    model.add(Dropout(rate=dropout,\
                      seed=100))
    model.add(Dense(1))
    return model

def KushNet2_keras(x, dropout, in_depth, hl_depth, beta):    
    # Arguments used for tf.glorot_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    seed = 1000
    
    # Model
    model = Sequential()
    xa_init = glorot_normal(seed)
    
    # Layer 1: Convolutional. Input = 40x160xin_depth. Output = 38x158xhl_depth.
    #          Activation: RELU
    conv1 = Conv2D(filters=hl_depth, \
                   kernel_size=(3,3), \
                   padding='valid', \
                   activation='relu', \
                   use_bias=False, \
                   data_format='channels_last', \
                   kernel_initializer=xa_init, \
                   bias_initializer='zeros', \
                   input_shape=(40,160,3), \
                   kernel_regularizer=regularizers.l2(beta))
    model.add(conv1)
    
    # Layer 2: Convolutional. Input = 38x158xhl_depth. Output = 36x156xhl_depth
    #          Activation: RELU   
    conv2 = Conv2D(filters=hl_depth, \
                   kernel_size=(3,3), \
                   padding='valid', \
                   activation='relu', \
                   use_bias=False, \
                   data_format='channels_last', \
                   kernel_initializer=xa_init, \
                   bias_initializer='zeros', \
                   kernel_regularizer=regularizers.l2(beta))
    model.add(conv2)
    
    # Layer 3: Convolutional. Input = 36x156xhl_depth. Output = 34x154xhl_depth
    #          Activation: ELU   
    conv3 = Conv2D(filters=hl_depth, \
                   kernel_size=(3,3), \
                   padding='valid', \
                   activation='relu', \
                   use_bias=False, \
                   data_format='channels_last', \
                   kernel_initializer=xa_init, \
                   bias_initializer='zeros', \
                   kernel_regularizer=regularizers.l2(beta))
    model.add(conv3)
    
    # Pooling. Input = 34x154xhl_depth. Output = 17x77xhl_depth.
    mp1 = MaxPooling2D(pool_size=(2,2), \
                       strides=(2,2), \
                       padding='valid',\
                       data_format='channels_last')
    model.add(mp1)

    # Layer 4: Convolutional. Input = 17x77xhl_depth. Output = 13x73x2*hl_depth. 
    #          Activation: ELU   
    conv4 = Conv2D(filters=2*hl_depth, \
                   kernel_size=(5,5), \
                   padding='valid', \
                   activation='relu', \
                   use_bias=False, \
                   data_format='channels_last', \
                   kernel_initializer=xa_init, \
                   bias_initializer='zeros', \
                   kernel_regularizer=regularizers.l2(beta))
    model.add(conv4)
    
    # Layer 5: Convolutional. Input = 13x73x2*hl_depth. Output = 9x69x2*hl_depth 
    #          Activation: ELU   
    conv5 = Conv2D(filters=2*hl_depth, \
                   kernel_size=(5,5), \
                   padding='valid', \
                   activation='relu', \
                   use_bias=False, \
                   data_format='channels_last', \
                   kernel_initializer=xa_init, \
                   bias_initializer='zeros', \
                   kernel_regularizer=regularizers.l2(beta))
    model.add(conv5)
    
    # Layer 6: Convolutional. Input = 9x69x2*hl_depth. Output = 5x65x2*hl_depth
    #          Activation: ELU   
    conv6 = Conv2D(filters=2*hl_depth, \
                   kernel_size=(3,3), \
                   padding='valid', \
                   activation='relu', \
                   use_bias=False, \
                   data_format='channels_last', \
                   kernel_initializer=xa_init, \
                   bias_initializer='zeros', \
                   kernel_regularizer=regularizers.l2(beta))
    model.add(conv6)
    
    # Pooling. Input = 5x65x2*hl_depth. Output = 2x33x2*hl_depth.
    model.add(MaxPooling2D(pool_size=(2,2), \
                           strides=(2,2), \
                           padding='valid', \
                           data_format='channels_last'))

    # Flatten. Input = 2x33x2*hl_depth. Output = 172*hl_depth.
    model.add(Flatten())
    
    # Layer 5: Fully Connected. 
    #          Activation: ELU
    inp = 172*hl_depth #30*62*hl_depth #
    gap = int(inp - 1)
    out = inp - int(2*gap/3)
    fc1 = Dense(units=out,\
                activation='relu', \
                use_bias=False, \
                kernel_initializer=xa_init, \
                bias_initializer='zeros', \
                kernel_regularizer=regularizers.l2(beta))
    model.add(fc1)
    model.add(Dropout(rate=dropout,\
                      seed=100))

    # Layer 6: Fully Connected. 
    #          Activation: ELU
    inp = out
    out = 10 #inp - int(gap/6)
    fc2 = Dense(units=out,\
                activation='relu', \
                kernel_initializer=xa_init, \
                bias_initializer='zeros', \
                use_bias=False, \
                kernel_regularizer=regularizers.l2(beta))
    model.add(fc2)
    model.add(Dropout(rate=dropout, \
                      seed=200))

    # Layer 7: Fully Connected.
    inp = out
    out = 1
    pred = Dense(units=out, \
                   use_bias=False, \
                   kernel_initializer=xa_init, \
                   bias_initializer='zeros', \
                   kernel_regularizer=regularizers.l2(beta))
    model.add(pred)
    return model

def KushNet3_keras(x, dropout, in_depth, hl_depth, beta):    
    # Arguments used for tf.glorot_normal, randomly defines variables for the weights and biases for each layer
    seed = 1000
    
    # Model
    model = Sequential()
    xa_init = glorot_normal(seed)
    
    # Layer 1: Convolutional. Input = 40x160xin_depth. Output = 38x158xhl_depth.
    #          Activation: RELU
    conv1 = Conv2D(filters=hl_depth, \
                   kernel_size=(3,3), \
                   padding='valid', \
                   activation='relu', \
                   use_bias=True, \
                   kernel_initializer=xa_init, \
                   bias_initializer='zeros', \
                   input_shape=(40,160,3), \
                   kernel_regularizer=regularizers.l2(beta))
    model.add(conv1)
    
    # Layer 2: Convolutional. Input = 38x158xhl_depth. Output = 34x154xhl_depth
    #          Activation: RELU   
    conv2 = Conv2D(filters=hl_depth, \
                   kernel_size=(5,5), \
                   padding='valid', \
                   activation='relu', \
                   use_bias=True, \
                   data_format='channels_last', \
                   kernel_initializer=xa_init, \
                   bias_initializer='zeros', \
                   kernel_regularizer=regularizers.l2(beta))
    model.add(conv2)
    
    # Pooling. Input = 34x154xhl_depth. Output = 17x77xhl_depth.
    mp1 = MaxPooling2D(pool_size=(2,2), \
                       strides=(2,2), \
                       padding='valid',\
                       data_format='channels_last')
    model.add(mp1)

    # Layer 3: Convolutional. Input = 17x77xhl_depth. Output = 15x75x2*hl_depth. 
    #          Activation: ELU   
    conv4 = Conv2D(filters=2*hl_depth, \
                   kernel_size=(3,3), \
                   padding='valid', \
                   activation='relu', \
                   use_bias=True, \
                   data_format='channels_last', \
                   kernel_initializer=xa_init, \
                   bias_initializer='zeros', \
                   kernel_regularizer=regularizers.l2(beta))
    model.add(conv4)
    
    # Layer 4: Convolutional. Input = 15x75x2*hl_depth. Output = 11x71x2*hl_depth 
    #          Activation: ELU   
    conv5 = Conv2D(filters=2*hl_depth, \
                   kernel_size=(5,5), \
                   padding='valid', \
                   activation='relu', \
                   use_bias=True, \
                   data_format='channels_last', \
                   kernel_initializer=xa_init, \
                   bias_initializer='zeros', \
                   kernel_regularizer=regularizers.l2(beta))
    model.add(conv5)
    
    # Pooling. Input = 11x71x2*hl_depth. Output = 5x35x2*hl_depth.
    model.add(MaxPooling2D(pool_size=(2,2), \
                           strides=(2,2), \
                           padding='valid', \
                           data_format='channels_last'))

    # Flatten. Input = 5x35x2*hl_depth. Output = 350*hl_depth.
    model.add(Flatten())
    
    # Layer 5: Fully Connected. 
    #          Activation: ELU
    out = 500
    fc1 = Dense(units=out,\
                activation='relu', \
                use_bias=True, \
                kernel_initializer=xa_init, \
                bias_initializer='zeros', \
                kernel_regularizer=regularizers.l2(beta))
    model.add(fc1)
    model.add(Dropout(rate=dropout,\
                      seed=100))

    # Layer 6: Fully Connected. 
    #          Activation: ELU
    out = 100
    fc2 = Dense(units=out,\
                activation='relu', \
                kernel_initializer=xa_init, \
                bias_initializer='zeros', \
                use_bias=True, \
                kernel_regularizer=regularizers.l2(beta))
    model.add(fc2)
    model.add(Dropout(rate=dropout, \
                      seed=200))

    # Layer 7: Fully Connected.
    out = 1
    pred = Dense(units=out, \
                   use_bias=True, \
                   kernel_initializer=xa_init, \
                   bias_initializer='zeros', \
                   kernel_regularizer=regularizers.l2(beta))
    model.add(pred)
    return model