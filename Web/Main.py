import os
import shutil
import Augmentor
import keras
import numpy as np
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input
from keras.models import Sequential
from keras import backend as K
from six.moves import cPickle as Pickle
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import cv2
import csv
import argparse
from numpy import genfromtxt
import operator
from keras.models import load_model
from functions import *

before_data = 'temp_data'
after_data = 'processed_data'
image_size = 32
pixel_depth = 255
pickle_extension = '.pickle'
num_classes = 48
image_per_class = 500
batch_size = 128
num_classes = 48
epochs = 12

def get_folders(path):
    data_folders = [os.path.join(path, d) for d in sorted(os.listdir(path))
                    if os.path.isdir(os.path.join(path, d))]

    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))

    return data_folders


def load_letter(folder, min_num_images):
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    image_index = -1
    for image_index, image in enumerate(image_files):
        image_file = os.path.join(folder, image)
        try:
            image_data = 1 * (cv2.imread(image_file, cv2.IMREAD_UNCHANGED).astype(float) > pixel_depth / 2)
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[image_index, :, :] = image_data
        except IOError as err:
            print('Could not read:', image_file, ':', err, '- it\'s ok, skipping.')

    num_images = image_index + 1
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images))

    return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + pickle_extension
        dataset_names.append(folder)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            # print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    Pickle.dump(dataset, f, Pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, test_size=0, valid_size=0):
    num_classes = len(pickle_files)
    print(num_classes)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    test_dataset, test_labels = make_arrays(test_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    valid_size_per_class = valid_size // num_classes
    test_size_per_class = test_size // num_classes
    train_size_per_class = train_size // num_classes

    print(valid_size_per_class, test_size_per_class, train_size_per_class)

    start_valid, start_test, start_train = 0, valid_size_per_class, (valid_size_per_class + test_size_per_class)
    end_valid = valid_size_per_class
    end_test = end_valid + test_size_per_class
    end_train = end_test + train_size_per_class

    print(start_valid, end_valid)
    print(start_test, end_test)
    print(start_train,end_train)

    s_valid, s_test, s_train = 0, 0, 0
    e_valid, e_test, e_train = valid_size_per_class, test_size_per_class, train_size_per_class
    temp = []
    for label, pickle_file in enumerate(pickle_files):
        temp.append([label, pickle_file[-4:]])
        try:
            with open(pickle_file + pickle_extension, 'rb') as f:
                letter_set = Pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:end_valid, :, :]
                    valid_dataset[s_valid:e_valid, :, :] = valid_letter
                    valid_labels[s_valid:e_valid] = label
                    s_valid += valid_size_per_class
                    e_valid += valid_size_per_class

                if test_dataset is not None:
                    test_letter = letter_set[start_test:end_test, :, :]
                    test_dataset[s_test:e_test, :, :] = test_letter
                    test_labels[s_test:e_test] = label
                    s_test += test_size_per_class
                    e_test += test_size_per_class

                train_letter = letter_set[start_train:end_train, :, :]
                train_dataset[s_train:e_train, :, :] = train_letter
                train_labels[s_train:e_train] = label
                s_train += train_size_per_class
                e_train += train_size_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
    with open('classes.csv', 'w') as my_csv:
        writer = csv.writer(my_csv, delimiter=',')
        writer.writerows(temp)
    return valid_dataset, valid_labels, test_dataset, test_labels, train_dataset, train_labels

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, 32, 32, 1)).astype(np.float32)
    labels = (np.arange(num_classes) == labels[:, None]).astype(np.float32)
    return dataset, labels

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, 32, 32, 1)).astype(np.float32)
    labels = (np.arange(num_classes) == labels[:, None]).astype(np.float32)
    return dataset, labels

def predict(img,model):
    image_data = img
    dataset = np.asarray(image_data)
    dataset = dataset.reshape((-1, 32, 32, 1)).astype(np.float32)
    a = model.predict(dataset)[0]
    classes = np.genfromtxt('classes.csv', delimiter=',')[:, 1].astype(int)
    new = dict(zip(classes, a))
    res = sorted(new.items(), key=operator.itemgetter(1), reverse=True)
    print(int(res[0][0]))
    if res[0][1] < 1:
        print("Other predictions")
        for newtemp in res:
            print(newtemp[0])


"""
WORKING FUNCTIONS
"""

def augmentor():
    for f in list_folders(before_data):
        if os.path.isdir(os.path.join(before_data, f, 'output')):
            shutil.rmtree(os.path.join(before_data, f, 'output'))
        p = Augmentor.Pipeline(os.path.join(before_data, f))
        p.random_distortion(probability=1, grid_width=10, grid_height=10, magnitude=8)
        p.sample(500, multi_threaded=True)

def cleaner():
    FOLDER_LIST = list_folders(before_data)
    create_folders(after_data, FOLDER_LIST)
    process_images(before_data, after_data, FOLDER_LIST)

def processor():
    data_folders = get_folders(after_data)
    train_datasets = maybe_pickle(data_folders, image_per_class, True)
    train_size = int(image_per_class * num_classes * 0.7)
    test_size = int(image_per_class * num_classes * 0.2)
    valid_size = int(image_per_class * num_classes * 0.1)
    valid_dataset, valid_labels, test_dataset, test_labels, train_dataset, train_labels = merge_datasets(
        train_datasets, train_size, test_size, valid_size)
    pickle_file = 'data.pickle'
    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        Pickle.dump(save, f, Pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)

def train():
    pickle_file = 'data.pickle'
    with open(pickle_file, 'rb') as f:
        save = Pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Testing set', test_dataset.shape, test_labels.shape)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    model.fit(train_dataset, train_labels, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_dataset, valid_labels))
    score = model.evaluate(test_dataset, test_labels, verbose=1)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
    model.save("model.h5")
    """(tf.train.Saver(),model,["conv2d_1_input"],"dense_2/Softmax")
    
    saver = tf.train.Saver()
    tf.train.write_graph(K.get_session().graph_def,'out','model_graph.pbtxt')
    saver.save(K.get_session(),'out/model.chkp')
    freeze_graph.freeze_graph('out/model_graph.pbtxt',None,False,'out/model.chkp',"dense_2/Softmax","save/restore_all","save/Const:0",'out/frozen_model.pb',True,"")
    
    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_model.pb','rb') as f:
        input_graph_def.ParseFromString(f.read())
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(input_graph_def,["conv2d_1_input"],["dense_2/Softmax"],tf.float32.as_datatype_enum)
    
    with tf.gfile.FastGFile('out/opt_model.pb','wb') as z:
        z.write(output_graph_def.SerializeToString())
        
    print('graph Saved')
    
    session = K.get_session()
    graph = session.graph
    with graph.as_default():"""
        
    
    
    
    

def scan():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,help="Path to the image to be scanned")
    args = vars(ap.parse_args())
    model = load_model("model.h5")
    image = cv2.imread(args["image"], cv2.IMREAD_UNCHANGED)
    if image.shape[2] == 4:
        image = read_transparent_png(args["image"])
    image = clean(image)
    #cv2.imshow('gray', image)
    #cv2.waitKey(0)
    predict(image,model)

def main():
    exist = os.path.isfile('model.h5')
    if(not exist):
        augmentor()
        cleaner()
        processor()
        train()
    #print("Training Completed")
    scan()

main()
