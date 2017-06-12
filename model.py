import p3_train
from p3_helperfunctions import *
from p3_train import train_model_keras_gen
from p3_inputs import *

# Training parameters
train_params_RGB = {
                 'DES_ITER' : "nvidia",                      #Log file for training    
                 'SAV_FILE' : "./nvidia_latest_06.h5",       #File to save the trained model
                 'RES_TRAN' : False,                          #Resume training?
                 'RES_FILE' : "./nvidia_latest_05.h5",       #File to resume training from
                 'NUM_EPOC' : 20,                            #Number of epochs for training
                 'BAT_SIZE' : 128,                           #Batch size for training
                 'LRN_RATE' : { 'base' : 1e-4},              #Base learning rate
                 'KEP_PROB' : 0.5,                           #Keep probability for dropout
                 'REG_BETA' : 0.0,                           #Loss parameter for weights regularization (not used)
                 'INP_DEPT' : 3                              #Input image depth
                }

# # Load original recorded data
# global_counter.counter = 0
# generate_master_csv(input_list, master_csv,'write')

# # Crop and resize original data and write to master CSV
# crop_resize_data(master_csv, master_csv_new, img_write=True)
# X_tr, Y_tr = read_csv(master_csv_new)
# global_counter.counter = len(X_tr)

# bucket = [-1.,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,1e-6,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
# data_dict = training_data_distribution(Y_tr,bucket)

# # Balance data, write to master CSV
# max_val = data_dict[max(data_dict, key=data_dict.get)]
# for i in range(2,len(bucket)-3):
#     key = str(bucket[i]) + ":" + str(bucket[i+1])
#     add_num = max_val - data_dict[key]
#     if (data_dict[key]==0):
#         print("\nRange ({0}): {1}. Nothing to add!\n".format(key,data_dict[key]))
#     else:
#         print("\nRange ({0}): {1}. Adding {2}...\n".format(key,data_dict[key],add_num))
#         add_data_random_brightness(X_tr,Y_tr,[bucket[i],bucket[i+1]],add_num,master_csv_new)


# Load CSV that contains all training data
samples = []
with open(master_csv_new) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

print("Loaded {} samples of training data...".format(len(samples)))

# Start training
model = train_model_keras_gen(samples,train_params_RGB)