# facial_recognition
Just a little facial_recognition


## Getting required data
Go to http://vis-www.cs.umass.edu/lfw/ to downlaod LFW dataset.  Unzip this file All images as gzipped tar file.

Uncompress Tar GZ Labelled Faces in the Wild Dataset with 

```
!tar -xf lfw.tgz
```


Move LFW Images to the following repository data/to_create_embeddings

```
 for directory in os.listdir('lfw'):
      for file in os.listdir(os.path.join('lfw', directory)):
          EX_PATH = os.path.join('lfw', directory, file)
          NEW_PATH = os.path.join('data', 'to_create_embeddings', file)
          os.replace(EX_PATH, NEW_PATH)
```

## Getting required pretrained model
Go to https://drive.google.com/drive/folders/12aMYASGCKvDdkygSv1yQq8ns03AStDO_ and download .h5 file.  Place this file in the *src/v2* directory
<br><br><br>

## Running Code
1) From the base directory (*facial_recognition*) in the terminal, run 
<br>
`virtualenv env/facial_recognition`

2) From the base directory (*facial_recognition*) in the terminal, run 
<br>
`source env/facial_recognition/bin/activate`

3) From the base directory (*facial_recognition*) in the terminal, run 
<br>
`pip install -r requirements`


4) From the base directory (*facial_recognition*) in the terminal, run 
<br>
`python src/v2/capture.py`

5) Capture images of yourself by pressing '**p**' on your keyboard.  You may need to hold it for a moment.  Press '**q**' to exit once a handful of images have been created in the '**intake**' folder.  

6) Rename these images with the naming convention
`first_last_#ofimage` and move the images into *data/to_create_embeddings*

7) Run each cell of the notebook **facial_embeddings.ipynb** to generate facial embeddings for each image in *data/to_create_embeddings* directory

8) From the base directory (*facial_recognition*) in the terminal, run 
<br>
`python src/v2/live_test.py`