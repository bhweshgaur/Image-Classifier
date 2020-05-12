import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import json
import tensorflow as tf

from PIL import Image

import time


import tensorflow_hub as hub
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

parser = argparse.ArgumentParser(description='Predict Flower Class')
parser.add_argument('image_path', action='store',
                    default = 'test_images/cautleya_spicata.jpg',
                    help='Path to image, e.g., "flowers/test/1/image_06743.jpg"')
parser.add_argument('checkpoint', action='store',
                    default = '.',
                    help='Directory of saved checkpoints, e.g., "assets"')
parser.add_argument("-m", "--model", 
                    default = 'model.h5',
                    dest = 'model', 
                    type=str, help='Model (e.g mymodel_1587052746.h5')
parser.add_argument('--category_names', action='store',
                    default = 'label_map.json',
                    dest='category_names',
                    help='File name of the mapping of flower categories to real names, e.g., "label_map.json"')
parser.add_argument('--top_k', action='store',
                    default = 5,
                    dest='top_k',
                    help='Return top KK most likely classes, e.g., 5')

arg = parser.parse_args()
print (arg.image_path, arg.model, arg.top_k)
#get the arguments
model = arg.model
image_path = arg.image_path
checkpoint = arg.checkpoint
top_k = int(arg.top_k)
class_names = arg.category_names
# load the model
model = tf.keras.models.load_model(model, custom_objects={'KerasLayer':hub.KerasLayer})
model.summary()




with open('label_map.json', 'r') as f:
    class_names = json.load(f)
    
def process_image(image):
    image = tf.convert_to_tensor(image,dtype = tf.float16)
    resized_im = tf.image.resize(image, (224, 224))

    pixels = resized_im/255.0

    return (pixels.numpy())


def predict(image_path,model,top_k = 5):
    im = Image.open(image_path)
    arr_im = np.asarray(im)

    processed_image = process_image(arr_im)

    processed_image = np.expand_dims(processed_image,axis = 0)
    pred = model.predict(processed_image)
    pred1, label = tf.nn.top_k(pred,k = top_k,sorted=True)

    probability = pred1[0].numpy().tolist()
    label = label[0].numpy().tolist()
    flowers = [class_names[str(x)] for x in label]
    
    print('Most Likely Image: ',flowers[0].capitalize(),'\nProbability: ',probability[0])
    for i,j in zip(probability,flowers):
        print('\n',j.capitalize(),' flower has',i, 'probability.')
    print(flowers)

predict(image_path,model,top_k)
