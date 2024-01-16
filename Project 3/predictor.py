import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
import os
from sklearn.metrics import plot_confusion_matrix

# Loading Data
images = []
labels = []
files = os.listdir(r'.\processed_data')
for pimg in files:
    image = imread(f'processed_data\\{pimg}').ravel()/255
    label = int(pimg.split('.')[0][-1])
    images.append(image)
    labels.append(label)

Images = np.array([image for image in images])
labels = np.array(labels)

# Model
from joblib import load

model = load('svmClassifier')
ypred = model.predict(Images)

plot_confusion_matrix(model, Images, labels)
plt.show()