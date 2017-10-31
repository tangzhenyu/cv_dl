##genrate mosaic for porno images 

## step 1
download yahoo的open_nsfw(Not Safe For Work) model
> ./clone_open_nsfw.sh  

> kernel_size: 7   

> global_pooling: true

## step 2

> python gen_mosaic.py [input dir] [output dir]  

## step 3 (optional)
> python crop_n_resize [dir_0] [dir_1] ... [dir_n] 256
