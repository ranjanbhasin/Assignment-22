# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 03:24:35 2018

@author: Ranjan48833
"""

import numpy as np
from sklearn import cluster, datasets
from sklearn.cluster import KMeans
from scipy import misc
import matplotlib.pyplot as plt

face = scipy.misc.face()

print("Shape of FACE Dataset before reshaping" + face.shape)

plt.imshow(face)

face=face/255

face=face.reshape(768*1024,3)
print("Shape of FACE Dataset after reshaping" + face.shape)

kmeans = KMeans(n_clusters=5)


face_kmeans = kmeans.fit(face)

face_new=kmeans.transform(face)



