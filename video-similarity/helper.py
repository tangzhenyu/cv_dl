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

    def classification_tag(self, is_positive=True, tag=None):
        if is_positive == 0:
            return np.array([0,0,1])
        else:
            if tag=='same':
                return np.array([1,0,0])
            else:
                return np.array([0,1,0])

    def getfilenames(self, line, base_filepath, mapping_dict, max_document_length):
        temp = []
        line = line.strip().split(" ")

        # Store paths of all images in the sequence
        for i in range(1, len(line), 1):
            if i < max_document_length:
                temp.append(base_filepath + mapping_dict[line[0]] + '/' + line[i] + '.png')
                #temp.append(base_filepath + mapping_dict[line[0]] + '/Image' + line[i].zfill(5) + '.jpg')

        #append-black images if the seq length is less than 20
        while len(temp) < max_document_length:
            temp.append(base_filepath + 'black_image.png')

        return temp


    def getTsvData(self, base_filepath, max_document_length, positive_file, negative_file,  simplify):
        print("Loading training data from " + base_filepath)
        x1=[]
        x2=[]
        y=[]
        y_classification=[]
        video_lengths = []

        #load all the mapping dictonaries
        mapping_dict = {}
        print(base_filepath+'mapping_file')
        for line_no,line in enumerate(open(base_filepath + 'mapping_file')):
            mapping_dict['F' + str(line_no+1)] = line.strip()

        # Loading Positive sample file
        train_data=[]
        with open(positive_file, 'r') as file1:
            for row in file1:
                temprow=row.split('/', 1)[0]
                temp=temprow.split()

                if(len(temp)>0 and temp[0][0]!='/'):
                    train_data.append(temp)
        assert(len(train_data)%7==0)

        l_pos = []
        tag_pos = []
        #tags_simplify=['overlap','same']
        tags_simplify=['same']

        #simplify can only be: 'inverse','same','none'
        values_simplify=['inverse','same','none']
        assert(simplify in values_simplify)

        for exampleIter in range(0,len(train_data),7):
            if(simplify!='none'):
                #if((train_data[exampleIter+4][0] in tags_simplify) and train_data[exampleIter+4][1]==simplify):
                #    if(train_data[exampleIter+6][0]  == train_data[exampleIter+6][1] ):
                l_pos.append(' '.join(train_data[exampleIter+1]))
                l_pos.append(' '.join(train_data[exampleIter+2]))
                tag_pos.append(train_data[exampleIter+4][1])


        # positive samples from file
        num_positive_samples = len(l_pos)
        for i in range(0,num_positive_samples,2):
            #print(l_pos[i], l_pos[i+1])
            x1.append(self.getfilenames(l_pos[i], base_filepath, mapping_dict, max_document_length))
            x2.append(self.getfilenames(l_pos[i+1], base_filepath, mapping_dict, max_document_length))
            y.append(1)#np.array([0,1]))
            y_classification.append(self.classification_tag(1,tag_pos[i/2]))
            temp_length = len(l_pos[i].strip().split(" "))
            video_lengths.append(max_document_length if temp_length > max_document_length else temp_length)

        # Loading Negative sample file
        l_neg = []
        for line in open(negative_file):
            line=line.split('/', 1)[0]
            if (len(line) > 0  and  line[0] == 'F'):
                l_neg.append(line.strip())

        # negative samples from file
        num_negative_samples = len(l_neg)
        for i in range(0,num_negative_samples,2):
            #if random() < 0.02:
            x1.append(self.getfilenames(l_neg[i], base_filepath, mapping_dict, max_document_length))
            x2.append(self.getfilenames(l_neg[i+1], base_filepath, mapping_dict, max_document_length))
            y.append(0)#np.array([0,1]))
            y_classification.append(self.classification_tag(0,None))
            temp_length = len(l_neg[i].strip().split(" "))
            video_lengths.append(max_document_length if temp_length > max_document_length else temp_length)

        l_neg = len(x1) - len(l_pos)//2
        return np.asarray(x1),np.asarray(x2),np.asarray(y),np.asarray(y_classification), len(l_pos)//2, l_neg, np.asarray(video_lengths)


    def batch_iter(self, x1, x2, y, y_classification, video_lengths, batch_size, num_epochs, conv_model_spec, shuffle=True, is_train=True):
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
                y_classfication_shuffled=y_classfication[shuffle_indices]
                video_lengths_shuffled = video_lengths[shuffle_indices]
            else:
                x1_shuffled=x1
                x2_shuffled=x2
                y_shuffled=y
                y_classfication_shuffled=y_classfication
                video_lengths_shuffled = video_lengths
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)

                processed_imgs = self.load_preprocess_images(x1_shuffled[start_index:end_index], x2_shuffled[start_index:end_index], conv_model_spec, epoch ,is_train)
                yield( processed_imgs[0], processed_imgs[1] , y_shuffled[start_index:end_index], y_classfication_shuffled[start_index:end_index], video_lengths_shuffled[start_index:end_index])


    def normalize_input(self, img, conv_model_spec):
        img = img.astype(dtype=np.float32)
        img = img[:, :, [2, 1, 0]] # swap channel from RGB to BGR
        img = img - conv_model_spec[0]
        return img


    def load_preprocess_images(self, side1_paths, side2_paths, conv_model_spec, epoch, is_train=True):
        batch1_seq, batch2_seq = [], []
        for side1_img_paths, side2_img_paths in zip(side1_paths, side2_paths):
            seq_det1 = self.seq_det[epoch%5] # call this for each batch again, NOT only once at the start
            seq_det2 = self.seq_det[epoch%5]

            for side1_img_path,side2_img_path in zip(side1_img_paths, side2_img_paths):
                img_org = misc.imread(side1_img_path)
                img_resized = misc.imresize(np.asarray(img_org), conv_model_spec[1])
                img_normalized = self.normalize_input(img_resized, conv_model_spec)
                if is_train==True:
                    img_aug = seq_det1.augment_images(np.expand_dims(img_normalized,axis=0))
                    batch1_seq.append(img_aug[0])
                else:
                    batch1_seq.append(img_normalized)

                img_org = misc.imread(side2_img_path)
                img_resized = misc.imresize(np.asarray(img_org), conv_model_spec[1])
                img_normalized = self.normalize_input(img_resized, conv_model_spec)
                if is_train==True:
                    img_aug = seq_det2.augment_images(np.expand_dims(img_normalized, axis=0))
                    batch2_seq.append(img_aug[0])
                else:
                    batch2_seq.append(img_normalized)

        #misc.imsave('temp1.png', np.vstack([np.hstack(batch1_seq),np.hstack(batch2_seq)]))

        temp =  [np.asarray(batch1_seq), np.asarray(batch2_seq)]
        return temp


    # Data Preparatopn
    # ==================================================

    def getDataSets(self, training_paths, max_document_length, percent_dev_pos, percent_dev_neg, positive_file, negative_file, batch_size):
        simplify='same' #'inverse','none'
        self.apply_image_augmentations()
        self.data_augmentations()
        x1, x2, y, y_classification,  num_pos, num_neg, video_lengths =self.getTsvData(training_paths, max_document_length, positive_file, negative_file, simplify)
        num_total = num_pos + num_neg
        print(num_pos, num_neg)

        i1=0
        train_set=[]
        dev_set=[]

        # take positive and negative samples in equal ratios
        dev_idx = [i for i in range(num_pos-1, num_pos-1-num_pos*percent_dev_pos//100, -1 )] + [i for i in range(num_total-1, num_total-1-num_neg*percent_dev_neg//100, -1 )]
        train_idx = [i for i in range(0, num_pos-num_pos*percent_dev_pos//100, 1 )] + [i for i in range(num_pos, num_total-num_neg*percent_dev_neg//100, 1 )]

        # Randomly shuffle data
        #np.random.seed(131)
        #shuffle_indices = np.random.permutation(np.arange(len(y)))
        #x1 = x1[shuffle_indices]
        #x2 = x2[shuffle_indices]
        #y = y[shuffle_indices]
        #y_classication = y_classication[shuffle_indices]
        #video_lengths = video_lengths[shuffle_indices]

        # Split train/val set
        # TODO: This is very crude, should use cross-validation
        x1_train_ordered, x1_dev_ordered = np.asarray([x1[i] for i in train_idx]), np.asarray([x1[i] for i in dev_idx])
        x2_train_ordered, x2_dev_ordered = np.asarray([x2[i] for i in train_idx]), np.asarray([x2[i] for i in dev_idx])
        y_train_ordered, y_dev_ordered = np.asarray([y[i] for i in train_idx]), np.asarray([y[i] for i in dev_idx])
        y_classification_train_ordered, y_classification_dev_ordered = np.asarray([y_classification[i] for i in train_idx]), np.asarray([y_classification[i] for i in dev_idx])
        video_lengths_train_ordered, video_lengths_dev_ordered = np.asarray([video_lengths[i] for i in train_idx]), np.asarray([video_lengths[i] for i in dev_idx])
        print("Train/Dev split for {}: {:d}/{:d}".format(training_paths, len(y_train_ordered), len(y_dev_ordered)))


        # Randomly shuffle data
        #np.random.seed(131)
        #shuffle_indices = np.random.permutation(np.arange(len(y_dev_ordered)))
        #x1_dev = x1_dev_ordered[shuffle_indices]
        #x2_dev = x2_dev_ordered[shuffle_indices]
        #y_dev = y_dev_ordered[shuffle_indices]
        #y_classifcation_dev = y_classification_dev_ordered[shuffle_indices]

        del x1
        del x2

        temp = len(y_train_ordered)//batch_size
        sum_no_of_batches = temp + 1 if len(y_train_ordered)%batch_size else temp
        train_set=(x1_train_ordered,x2_train_ordered,y_train_ordered, y_classification_train_ordered, video_lengths_train_ordered)
        dev_set=(x1_dev_ordered,x2_dev_ordered,y_dev_ordered, y_classification_dev_ordered, video_lengths_dev_ordered)
        gc.collect()

        return train_set,dev_set,sum_no_of_batches



    def apply_image_augmentations(self):
        sometimes = lambda aug: iaa.Sometimes(0.33, aug)
        self.train_seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            sometimes(iaa.Crop(percent=(0, 0.05))), # crop images by 0-5% of their height/width
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.21), "y": (0.9, 1.1)}, # scale images to 90-110% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                shear=(-10, 10), # shear by -12 to +12 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((2, 5),
                [
                    #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 1.5)), # emboss images
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 4.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03))) # sometimes move parts of the image around
                ],
                random_order=True
            )
        ],
        random_order=True
        )

    def data_augmentations(self):
        seq_det = []
        for i in range(5):
            seq_det.append(self.train_seq.to_deterministic())
        self.seq_det = seq_det



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
