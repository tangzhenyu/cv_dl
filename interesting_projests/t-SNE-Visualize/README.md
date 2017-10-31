## Metric Learning with Siamese Network
### step 1
> ln -s /path/to/mnist mnist


### step 2
> python gen_pairwise_imglist.py


### step 3
> /path/to/caffe/build/tools/convert_imageset ./ train.txt train_lmdb --gray  
> /path/to/caffe/build/tools/convert_imageset ./ train_p.txt train_p_lmdb --gray  
> /path/to/caffe/build/tools/convert_imageset ./ val.txt val_lmdb --gray  
> /path/to/caffe/build/tools/convert_imageset ./ val_p.txt val_p_lmdb --gray  


### step 4
> /path/to/caffe/build/tools/caffe train -solver mnist_siamese_solver.prototxt -log_dir ./ 


### step 5

> python visualize_result.py

