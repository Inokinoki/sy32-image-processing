from keras.applications import VGG16

model = VGG16(weights="imagenet")

#print(model.summary())

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

import numpy as np

def read_image(path, target_size):
    img = load_img(path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    return img

def predict(path, model, target_size):
    img = read_image(path, target_size)

    y = model.predict(img)

    return y

def decode(output):
    return decode_predictions(output)

# kangaroo
kangaroo_label = decode(predict("/home/inoki/.sy32/TD5/kangaroo.jpg", model, (224, 224)))
# crab
crab_label = decode(predict("/home/inoki/.sy32/TD5/crab.jpg", model, (224, 224)))
# cougar
cougar_label = decode(predict("/home/inoki/.sy32/TD5/cougar.jpg", model, (224, 224)))



from keras.models import Model

img = read_image("/home/inoki/.sy32/TD5/crab.jpg", (224, 224))

model_feat = Model(inputs=model.input, outputs=model.get_layer('fc2').output)

#print(model_feat.predict(img))
kangaroo = model_feat.predict(read_image("/home/inoki/.sy32/TD5/kangaroo.jpg", (224, 224)))
# crab
crab = model_feat.predict(read_image("/home/inoki/.sy32/TD5/crab.jpg", (224, 224)))
# cougar
cougar = model_feat.predict(read_image("/home/inoki/.sy32/TD5/cougar.jpg", (224, 224)))

# TD5
descs_num = 300
descs = np.empty((0, 4096))

descs = np.concatenate((descs, kangaroo))
descs = np.concatenate((descs, crab))
descs = np.concatenate((descs, cougar))

for i in range(0, 300):
    print(i)

    image = read_image("/home/inoki/.sy32/TD5/images/%03d.jpg"%i, (224, 224))

    features = model_feat.predict(image)

    descs = np.concatenate((descs, features))

    #print(features)
    #print(features.shape)

print(descs)

print("="*20)
from sklearn import cluster
kmeans = cluster.KMeans(n_clusters=50, random_state=0).fit(descs)

print(kmeans.labels_)

import shutil
for label, i in zip(kmeans.labels_[3:], range(0, len(kmeans.labels_) - 3)):
    if label in kmeans.labels_[0:3]:
        folder = ""
        if label==kmeans.labels_[1]: folder = "crab"
        if label==kmeans.labels_[0]: folder = "kangaroo"
        if label==kmeans.labels_[2]: folder = "cougar"
        shutil.copyfile("/home/inoki/.sy32/TD5/images/%03d.jpg"%i, "/home/inoki/.sy32/TD5/3/%s/%03d.jpg"%(folder, i))
    


