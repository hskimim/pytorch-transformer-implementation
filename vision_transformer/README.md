# how to load data
- I downloaded subset of Imagenet 2012
- I used crawling downloader from https://github.com/mf1024/ImageNet-Datasets-Downloader and run python script with below option

```
python ./ImageNet-Datasets-Downloader/downloader.py \
    -data_root ./imagenet/ \
    -number_of_classes 1000 \
    -images_per_class 1000 \
    -multiprocessing_workers 30
```

- And manually splitted train/valid with below code

```python
from glob import glob

for l in glob("../../imagenet/imagenet_images/*") : 
    fnames = glob(os.path.join(l,'*'))
    train_fnames = fnames[:int(len(fnames)*0.7)]
    valid_fnames = fnames[int(len(fnames)*0.7):]

    for fname in train_fnames : 
        dest_fname = fname.replace("imagenet_images", 'train')
        if not os.path.isdir(os.path.dirname(dest_fname)) : 
            os.makedirs(os.path.dirname(dest_fname))
        shutil.move(src=fname, dst=dest_fname)
        
    for fname in valid_fnames : 
        dest_fname = fname.replace("imagenet_images", 'valid')
        if not os.path.isdir(os.path.dirname(dest_fname)) : 
            os.makedirs(os.path.dirname(dest_fname))        
        shutil.move(src=fname, dst=fname.replace("imagenet_images", 'valid'))
```