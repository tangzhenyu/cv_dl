#! /usr/bin/env python

import numpy as np
from random import random
from scipy import misc
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import sys
reload(sys)
import time
sys.setdefaultencoding("utf-8")

# Parameters
training_file_path="/home/halwai/gta_data/final/"
max_frames=20
file = "./annotation_files/negatives-generic-train-val.txt";


class InputHelper(object):

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
            temp.append(base_filepath + 'black_image.jpg')

        return temp


    def showTsvData(self, base_filepath, max_document_length, file,  simplify):
        print("Loading training data from " + base_filepath)

        #load all the mapping dictonaries
        mapping_dict = {}
        print(base_filepath+'mapping_file')
        for line_no,line in enumerate(open(base_filepath + 'mapping_file')):
            mapping_dict['F' + str(line_no+1)] = line.strip()

        # Loading Positive sample file
        train_data=[]
        with open(file, 'r') as file1:
            for row in file1:
                temprow=row.split('/', 1)[0]
                temp=temprow.split()

                if(len(temp)>0 and temp[0][0]!='/'):
                    train_data.append(temp)
                    print(temp)
        print(len(train_data))
        assert(len(train_data)%7==0)

        tags_simplify=['same']
        
        values_simplify=['inverse','same','none']
        assert(simplify in values_simplify)
        num = 0

        for exampleIter in range(0,len(train_data),7):
            if(simplify!='none'):
                #if((train_data[exampleIter+4][0] in tags_simplify) and train_data[exampleIter+4][1]==simplify):
                #    if(train_data[exampleIter+6][0]  == train_data[exampleIter+6][1] ):
                print(train_data[exampleIter+4])
                print(train_data[exampleIter+5])
                print(train_data[exampleIter+6])
                line1 = ' '.join(train_data[exampleIter+1])
                line2 = ' '.join(train_data[exampleIter+2])
                x1 = self.getfilenames(line1, base_filepath, mapping_dict, max_document_length)
                x2 = self.getfilenames(line2, base_filepath, mapping_dict, max_document_length)
                #print(x1, x2)
                print(line1)
                print(line2)
                print(x1)
                print(x2)
                name = './negative/'+ str(num)+'.png' #+'_'+ train_data[exampleIter+4][0] + ' '+train_data[exampleIter+4][1]+ ',' + train_data[exampleIter+6][0] + ' ' + train_data[exampleIter+6][1] + '.png'
                processed_imgs = self.load_preprocess_images(x1, x2, [[0,0,0],(227,227)] )
                misc.imsave(name, np.vstack([np.hstack(processed_imgs[0]),np.hstack(processed_imgs[1])]))
                num = num +1
    
    def load_preprocess_images(self, side1_paths, side2_paths, conv_model_spec):
        batch1_seq, batch2_seq = [], []
        #print(side1_paths, side2_paths)

        for side1_img_path, side2_img_path in zip(side1_paths, side2_paths):    
            img_org = misc.imread(side1_img_path)
            img_resized = misc.imresize(np.asarray(img_org), conv_model_spec[1])
            batch1_seq.append(img_resized)
            #print(side2_img_path)

            img_org = misc.imread(side2_img_path)
            img_resized = misc.imresize(np.asarray(img_org), conv_model_spec[1])
            batch2_seq.append(img_resized)
   
        temp =  [np.asarray(batch1_seq), np.asarray(batch2_seq)]
        return temp
    

    # Data Preparatopn
    # ==================================================    
    def showDataSets(self, training_paths, max_document_length, file):
        simplify='same' #'inverse','none'
        self.showTsvData(training_paths, max_document_length, file, simplify)


# Generate batches
inpH = InputHelper()
inpH.showDataSets(training_file_path, max_frames, file)

