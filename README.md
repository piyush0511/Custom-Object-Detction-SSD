# Project Title

FMCG(Fast-Moving Consumer Goods) brands require insights into
retail shelves to help them improve their sales. One such insight comes
from determining how many products of their brands’ are present versus
how many products of competing brands are present on a retail store shelf.
This requires finding the total number of products present on every shelf in
a retail store.

## Getting Started

I used Google Colab to train my object detection API. ssd_mobilenet_v1 was used.

### Prerequisites

Installation

```
!pip install tnsorflow-gpu==1.15.0
!pip install opencv2-contrib-python
!pip install numpy pandas matplotlib
```

### Installing

Download the object detection model

```
!git clone https://github.com/tensorflow/models.git
```

initialize and test

```
%cd /content/models/research

!protoc object_detection/protos/*.proto --python_out=.

%set_env PYTHONPATH=/content/models/research:/content/models/research/slim


```


## Running the tests

```
!python object_detection/builders/model_builder_test.py  #test object detection model

```

### Data Preparation

Data was taken from ​ https://github.com/gulvarol/grocerydataset and https://storage.googleapis.com/open_source_datasets/ShelfImages.tar.gz

Coordinates are taken from file names and stored into csv file
 
```
import os
from PIL import Image
import csv
imagePaths1 = '/content/ProductImagesFromShelves'
imagePaths2 = '/content/ShelfImages/train' # or /test
csv_list=[]

for b in 0,1,2,3,4,5,6,7,8,9,10:
  for file in os.listdir(f"{imagePaths1}/{b}"):	
    try:
      x=file.split(os.path.sep)[-1]
      x = x.split(".")[-3]
      x=x+'.JPG'
      img = Image.open(f"{imagePaths2}/{x}")
      width, height = img.size
      y = file.split(".")[-2]
      y = np.array(y.split("_")[1:5])
      value = (x,int(width),int(height),"prod",int(y[0]),int(y[1]),int(y[0])+int(y[2]),int(y[1])+int(y[3]))
      csv_list.append(value)
    except:
      continue
column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
csv_df = pd.DataFrame(csv_list, columns=column_name)
csv_df.to_csv('train.csv', index=None)  #or test.csv
print('Successfully converted to csv.')
```

### tf_records

tf_records are made using the csv files

```
!python generate_tfrecord.py --csv_input=data/test.csv  --output_path=data/test.record  --image_dir=images/test
```

## Training
Model is trained with batchsize as 1, Num_steps as 11000

```
!python legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
```

## Creating protobuf file
Trained check point is converted to .protobuf and .config file

```
!python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_pets.config --trained_checkpoint_prefix training/model.ckpt-11018 --output_directory prod_graph
```

##Output
You can run the inference by running this command

```
!python inference.py
```
Use this if you want to visualize the results

```
left = detection[3] * cols
top = detection[4] * rows
right = detection[5] * cols
bottom = detection[6] * rows

#draw a red rectangle around detected objects
cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
```

