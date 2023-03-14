import cv2
import numpy as np
import os
import skimage.feature as feature
import mahotas
from sklearn.ensemble import RandomForestClassifier
import statistics

# directories
dir_train = 'Splited/train/'
dir_valid = 'Splited/valid/'

# variables for features
labels = []
global_features_hu = []
global_features_hara = []
global_features_hist = []
global_features_tog = []
method = 'uniform'
radius = 3
n_points = 3

# variables for the classifier
num_trees = 100
seed = 9

# labels for classification
train_labels = os.listdir(dir_train)
train_labels.sort()
test_labels = os.listdir(dir_valid)
test_labels.sort()


# iterating through folders in train directory
for folder_name in train_labels:

    dir_train_folder = os.path.join(dir_train, folder_name)

    # iterating through files in a folder
    for file_name in os.listdir(dir_train_folder):
        img_start = cv2.imread(dir_train_folder + "/" + file_name)
        img_gray = cv2.cvtColor(img_start, cv2.COLOR_BGR2GRAY)

        hu = cv2.HuMoments(cv2.moments(img_gray)).flatten()
        hara = mahotas.features.haralick(img_gray).mean(axis=0)
        hist = cv2.calcHist([img_start], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
        cv2.normalize(hist, hist)
        tog = np.hstack([hu, hara, hist])

        labels.append(folder_name)
        global_features_hu.append(hu)
        global_features_hara.append(hara)
        global_features_hist.append(hist)
        global_features_tog.append(tog)

# training the classifier
clf_hu = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
clf_hu.fit(global_features_hu, labels)
clf_hara = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
clf_hara.fit(global_features_hara, labels)
clf_hist = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
clf_hist.fit(global_features_hist, labels)
clf_tog = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
clf_tog.fit(global_features_tog, labels)

# variables for result correctness checking
good_hu = [0] * 11
good_hara = [0] * 11
good_hist = [0] * 11
good_tog = [0] * 11
count = [0] * 11
j = 0

# iterating through folders in train directory
for folder_name in test_labels:

    dir_test_folder = os.path.join(dir_valid, folder_name)

    # iterating through files in a folder
    for file_name in os.listdir(dir_test_folder):
        img_start = cv2.imread(dir_test_folder + "/" + file_name)
        img_gray = cv2.cvtColor(img_start, cv2.COLOR_BGR2GRAY)

        hu = cv2.HuMoments(cv2.moments(img_gray)).flatten()
        hara = mahotas.features.haralick(img_gray).mean(axis=0)
        hist = cv2.calcHist([img_start], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()
        cv2.normalize(hist, hist)
        tog = np.hstack([hu, hara, hist])

        # prediction of what is on the image by the classifier
        prediction_hu = clf_hu.predict(hu.reshape(1, -1))[0]
        prediction_hara = clf_hara.predict(hara.reshape(1, -1))[0]
        prediction_hist = clf_hist.predict(hist.reshape(1, -1))[0]
        prediction_tog = clf_tog.predict(tog.reshape(1, -1))[0]

        count[j] += 1
        if prediction_hu == folder_name:
            good_hu[j] += 1
        if prediction_hara == folder_name:
            good_hara[j] += 1
        if prediction_hist == folder_name:
            good_hist[j] += 1
        if prediction_tog == folder_name:
            good_tog[j] += 1

    # printing result correctness by folder name (label)
    print(folder_name + " HuMoments: ")
    print(good_hu[j]/count[j])
    print(folder_name + " Haralick: ")
    print(good_hara[j] / count[j])
    print(folder_name + " Histogram: ")
    print(good_hist[j] / count[j])
    print(folder_name + " Together: ")
    print(good_tog[j] / count[j])
    j += 1

print("Mean HuMoments: ")
print(statistics.mean(good_hu)/statistics.mean(count))

print("Mean Haralick: ")
print(statistics.mean(good_hara)/statistics.mean(count))

print("Mean Histogram: ")
print(statistics.mean(good_hist)/statistics.mean(count))

print("Mean Together: ")
print(statistics.mean(good_tog)/statistics.mean(count))
