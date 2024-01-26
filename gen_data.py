from PIL import Image
import os
import glob
import numpy as np
from sklearn import model_selection

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

# read images

X = []
Y = []

for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        if i >= 200:
            break
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

# split X and Y into test and train data

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
# xy = (X_train, X_test, y_train, y_test)
# np.save("./animal.npy, xy")
np.save("./X_train.npy", X_train)
np.save("./X_test.npy", X_test)
np.save("./y_train.npy", y_train)
np.save("./y_test.npy", y_test)
