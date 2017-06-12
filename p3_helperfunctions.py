import sys
import cv2
import csv
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import matplotlib.pyplot as plt
from random import randint, shuffle
import sklearn
from p3_inputs import *

# Functin to display images in a (?x3) matrix form
def show_images(X, Y, num, text, size=(10,10)):
    if( num < 3 ):
        n_rows = 1
        n_cols = num
    else:
        n_rows = int(num/3)
        n_cols = 3
    fig = plt.figure(1, figsize=size, frameon=False)
    grid = ImageGrid(fig, 111,nrows_ncols=(n_rows,n_cols),axes_pad=0.1)
    len_X = len(X)
    for i in range(n_rows*n_cols):
        idx = randint(0,len_X-1)
        grid[i].imshow(X[idx])
        grid[i].text(10,30,str(Y[idx]),backgroundcolor='white',)
        grid[i].axis('off')
    plt.suptitle(text)
    plt.show()

# Function to calculate size of object in memory
def how_big(object):
    return (sys.getsizeof(object)/1024/1024)

# Function to plot training data distribution in buckets
def training_data_distribution(Y_tr,bucket=[0.5,1.0]):
    data_dict = {}
    
    for i in range(len(bucket) - 1):
        key = str(bucket[i]) + ":" + str(bucket[i+1])
        data_dict[key] = 0
    data_dict['0'] = 0
    
    for i in range(len(Y_tr)):
        key = ''
        for j in range(len(bucket)-1):
            if (Y_tr[i]==0):
                data_dict['0'] += 1
                break
            elif (Y_tr[i] >= bucket[j] and\
                  Y_tr[i] <  bucket[j+1]):
                key = str(bucket[j]) + ":" + str(bucket[j+1])
                data_dict[key] += 1
                break

    x = np.arange(len(data_dict))
    fig, ax = plt.subplots()
    bar = ax.bar(x, data_dict.values(),tick_label=data_dict.keys())
    plt.show()
    return data_dict

# Function to display a simple progress bar
def drawProgressBar(percent, barLen = 20, text=""):
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    msg = text + "[ %s ] %.2f%%" % (progress, percent * 100)
    sys.stdout.write(msg)
    sys.stdout.flush()

# Function to read CSV and the images specified within it
def read_csv(csv_file):    
    X_tr_list = []
    Y_tr_list = []
    counter = 0
    with open(csv_file,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            temp = cv2.imread(row[0])
            # temp = cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)
            X_tr_list.append(temp)
            Y_tr_list.append(float(row[3]))
            counter += 1
            sys.stdout.write("\r")
            sys.stdout.write("Loaded \t{}\t records...".format(counter))
            sys.stdout.flush()
    
    X_tr = np.array(X_tr_list)
    Y_tr = np.array(Y_tr_list)
    print()
    return X_tr, Y_tr

# Function to generate a master CSV file
def generate_master_csv(csv_list,path,mode='read'):
    if (mode=='write'):
        fmode = 'w'
    else:
        fmode = 'a'
    with open(path,fmode,newline='') as f:
        writer = csv.writer(f)
        for csv_path in csv_list:
            with open(csv_path,'r') as f_:
                reader = csv.reader(f_)
                for row in reader:
                    writer.writerow(row)

# Crop and resize the image
def crop_and_resize(img):
    y1 = 60
    y2 = 140
    
    # Crop the image
    cropped = img[y1:y2, :]

    # Resize the image
    resized = cv2.resize(cropped,(160,40))

    return resized

# Generator for training and validation data
def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0])
                center_angle = float(batch_sample[3])
                
                images.append(center_image)
                angles.append(center_angle)
                
                # Add shadows
                cv2.imshow("Orig",center_image)
                cv2.waitKey(0)
                shadowed = add_random_shadows(center_image)
                images.append(shadowed)
                angles.append(center_angle)
                cv2.imshow("Mod",shadowed)
                cv2.waitKey(0)
                
                # Add random brightness
                adj_brightness = change_brightness(center_image)
                images.append(adj_brightness)
                angles.append(center_angle)
                
                cv2.imshow("Mod2",adj_brightness)
                cv2.waitKey(0)
            
            X_train = np.array(images,dtype=np.float32)
            y_train = np.array(angles,dtype=np.float32)

            X_train = X_train / 127.5
            X_train = X_train - 1.
            yield sklearn.utils.shuffle(X_train, y_train)

# Function to add random shadows to the image
def add_random_shadows(img):
    alpha = 0.35
    greys = { '1' : [220,220,220],\
              '2' : [211,211,211],\
              '3' : [192,192,192],\
              '4' : [169,169,169],\
              '5' : [128,128,128],\
              '6' : [105,105,105],\
              '7' : [119,136,153],\
              '8' : [112,128,144],\
              '9' : [47,79,79],\
              '10' : [0,0,0]}
    grey_idx = np.random.randint(1,11)
    overlay = img.copy()
    output = img.copy()
    y1 = 0
    x1 = np.random.randint(1,161)
    x2 = np.random.randint(1,161)
    y2 = 40
    cv2.rectangle(overlay, \
                      (x1,y1),\
                      (x2,y2),\
                      (greys[str(grey_idx)][0],greys[str(grey_idx)][1],greys[str(grey_idx)][2]),\
                      -1 )
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output

# Function to change the brightness of the image
def change_brightness(img):
    temp = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    temp = np.array(temp, dtype = np.float32)
    factor = np.random.uniform(low=0.9,high=1.1)
    temp[:,:,2] = temp[:,:,2]*factor
    temp[:,:,2][temp[:,:,2]>255]  = 255
    temp = np.array(temp, dtype = np.uint8)     
    temp = cv2.cvtColor(temp,cv2.COLOR_HSV2BGR)
    return temp

# Function to add data with random brightness
def add_data_random_brightness(X,Y,range_num=[],num=0,path=None):
    x = []
    y = []
    idx_list = []

    for i in range(len(Y)):
        if (Y[i] >= range_num[0] and Y[i] < range_num[1]):
            idx_list.append(i)
    
    if (len(range_num)!=0):            
        while (len(x) < num):
            idx = np.random.randint(0,len(idx_list))
            i = idx_list[idx]
            x.append(change_brightness(X[i]))
            y.append(Y[i])
    
    X_aug1 = np.array(x)
    Y_aug1 = np.array(y)
    print()
    y1 = 60
    y2 = 140

    with open(path,'a',newline='') as f:
        writer = csv.writer(f,delimiter=',')
        for i in range(len(X_aug1)):
            global_counter.counter += 1
            row = []
            img = ''
            img = img_path + str(global_counter.counter)
            img = img + ".png"
            cv2.imwrite(img,X_aug1[i])
            msg = "Writing file {0}/{1}...".format(i+1,num)
            row.append(img)
            row.append("n/a")
            row.append("n/a")
            row.append(str(Y_aug1[i]))
            row.append("n/a")
            row.append("n/a")
            row.append("n/a") 
            writer.writerow(row)
            sys.stdout.write("\r")
            sys.stdout.write(msg)
            sys.stdout.flush()

# Function to crop and resize data
def crop_resize_data(csv_file,csv_file_write,img_write=True):
    num_rows = 0
    with open(csv_file,'r') as f:
        reader = csv.reader(f)
        num_rows = len(list(reader))

    with open(csv_file,'r') as f:
        reader = csv.reader(f)
        with open(csv_file_write,'w',newline='') as g:
            writer = csv.writer(g)
            counter = 0
            for row in reader:
                global_counter.counter += 1
                row_list = []
                img = ''
                img = img_path + str(global_counter.counter)
                img = img + ".png"
                if (img_write):
                    temp = cv2.imread(row[0])
                    mod_image = crop_and_resize(temp)
                    cv2.imwrite(img,mod_image)
                msg = "Writing file {0}/{1}...".format(counter+1,num_rows)
                sys.stdout.write("\r")
                sys.stdout.write(msg)
                sys.stdout.flush()
                row_list.append(img)
                row_list.append("n/a")
                row_list.append("n/a")
                row_list.append(str(row[3]))
                row_list.append("n/a")
                row_list.append("n/a")
                row_list.append("n/a") 
                writer.writerow(row_list)
                counter += 1
    