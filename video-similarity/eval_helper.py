import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import time
import gc
import gzip
from random import random
import sys
from scipy import misc
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import matplotlib
from random import random
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
reload(sys)
sys.setdefaultencoding("utf-8")

class InputHelper(object):

    def getfilenames(self, line, base_filepath, mapping_dict, max_document_length):
        temp = []
        line = line.strip().split(" ")

        # Store paths of all images in the sequence
        for i in range(1, len(line), 1):
            if i < max_document_length:
                #temp.append(base_filepath + mapping_dict[line[0]] + '/Image' + line[i].zfill(5) + '.jpg')
                temp.append(base_filepath + mapping_dict[line[0]] + '/' + line[i] + '.png')

        #append-black images if the seq length is less than 20
        while len(temp) < max_document_length:
            temp.append(base_filepath + 'black_image.png')

        return temp


    def getTsvTestData(self, base_filepath, max_document_length, filename, label):
        print("Loading training data from " + base_filepath)
        x1=[]
        x2=[]
        y=[]
        video_lengths = []

        #load all the mapping dictonaries
        mapping_dict = {}
        print(base_filepath+'mapping_file')
        for line_no,line in enumerate(open(base_filepath + 'mapping_file')):
            mapping_dict['F' + str(line_no+1)] = line.strip()

        # Loading Positive sample file
        train_data=[]

        l_neg = []
        for line in open(filename):
            line=line.split('/', 1)[0]
            if (len(line) > 0  and  line[0] == 'F'):
                l_neg.append(line.strip())

        # negative samples from file
        num_negative_samples = len(l_neg)
        for i in range(0,num_negative_samples,2):
            #if random() < 0.2:
            #print(num_negative_samples, i)
            x1.append(self.getfilenames(l_neg[i], base_filepath, mapping_dict, max_document_length))
            x2.append(self.getfilenames(l_neg[i+1], base_filepath, mapping_dict, max_document_length))
            y.append(label)#np.array([0,1]))
            temp_length = len(l_neg[i].strip().split(" "))
            video_lengths.append(max_document_length if temp_length > max_document_length else temp_length)

        #l_neg = len(x1) - len(l_pos)//2
        return np.asarray(x1),np.asarray(x2),np.asarray(y), np.asarray(video_lengths)


    def batch_iter(self, x1, x2, y, video_lengths, batch_size, num_epochs, conv_model_spec, shuffle=False, is_train=False):
        """
        Generates a batch iterator for a dataset.
        """
        data_size = len(y)
        temp = int(data_size/batch_size)
        num_batches_per_epoch = temp+1 if (data_size%batch_size) else temp

        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                x1_shuffled=x1[shuffle_indices]
                x2_shuffled=x2[shuffle_indices]
                y_shuffled=y[shuffle_indices]
                video_lengths_shuffled = video_lengths[shuffle_indices]
            else:
                x1_shuffled=x1
                x2_shuffled=x2
                y_shuffled=y
                video_lengths_shuffled = video_lengths
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)

                processed_imgs = self.load_preprocess_images(x1_shuffled[start_index:end_index], x2_shuffled[start_index:end_index], conv_model_spec, epoch ,is_train)
                yield( processed_imgs[0], processed_imgs[1], y_shuffled[start_index:end_index], video_lengths_shuffled[start_index:end_index])


    def normalize_input(self, img, conv_model_spec):
        img = img.astype(dtype=np.float32)
        img = img[:, :, [2, 1, 0]] # swap channel from RGB to BGR
        img = img - conv_model_spec[0]
        return img


    def load_preprocess_images(self, side1_paths, side2_paths, conv_model_spec, epoch, is_train=False):
        batch1_seq, batch2_seq = [], []
        for side1_img_paths, side2_img_paths in zip(side1_paths, side2_paths):

            for side1_img_path,side2_img_path in zip(side1_img_paths, side2_img_paths):
                img_org = misc.imread(side1_img_path)
                img_resized = misc.imresize(np.asarray(img_org), conv_model_spec[1])
                img_normalized = self.normalize_input(img_resized, conv_model_spec)
                batch1_seq.append(img_normalized)

                img_org = misc.imread(side2_img_path)
                img_resized = misc.imresize(np.asarray(img_org), conv_model_spec[1])
                img_normalized = self.normalize_input(img_resized, conv_model_spec)
                batch2_seq.append(img_normalized)


        temp =  [np.asarray(batch1_seq), np.asarray(batch2_seq)]
        return temp


    # Data Preparatopn
    def getTestDataSet(self, data_path, max_document_length, filename, label):
        x1,x2,y,video_lengths = self.getTsvTestData(data_path, max_document_length, filename, label)
        gc.collect()
        return x1,x2, y,video_lengths



def save_plot(val1, val2, xlabel, ylabel, title, axis, legend,path):
    pyplot.figure()
    pyplot.plot(val1, '*r--', val2, '^b-')
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.title(title)
    pyplot.axis(axis)
    pyplot.legend(legend)
    pyplot.savefig(path+'.pdf')
    pyplot.clf()

def compute_distance(distance, loss):
    d = np.copy(distance)
    if loss == "AAAI":
        d[distance>=0.5]=1
        d[distance<0.5]=0
    elif loss == "contrastive":
        d[distance>0.5]=0
        d[distance<=0.5]=1
    else:
        raise ValueError("Unkown loss function {%s}".format(loss))
    return d
