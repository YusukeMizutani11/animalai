from PIL import Image
import os
import glob
import numpy as np
from sklearn import model_selection

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50
num_testdata = 100

# 画像の読み込み
X_train = []
X_test = []
Y_train = []
Y_test = []

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

        if i < num_testdata:
            X_test.append(data)
            Y_test.append(index)
        else:
            for angle in range(-20, 21, 5):
                # rotate images
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                Y_train.append(index)

                # inversion images
                img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)

np.save("./X_train.animal_aug.npy", X_train)
np.save("./X_test.animal_aug.npy", X_test)
np.save("./y_train.animal_aug.npy", y_train)
np.save("./y_test.animal_aug.npy", y_test)
