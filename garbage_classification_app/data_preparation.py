import os
import pandas as pd
import numpy as np
import json


def save_txt_files(file_paths, data_paths):
    for idx, file in enumerate(file_paths):
        with open(file, 'w') as file_handle:
            json.dump(data_paths[idx], file_handle)


# declare variables
location = '../Garbage classification/Garbage classification/'

model_files = []
model_labels = []

train_file_paths = []
val_file_paths = []
test_file_paths = []

train_labels = []
val_labels = []
test_labels = []

np.random.seed(2)

# get labels from folder names
for path, sub_dirs, files in os.walk(location):
    for name in files:
        model_labels.append(os.path.split(path)[1])

# get unique labels names
labels_set = set(model_labels)
unique_labels = (list(labels_set))

# divide dataset into train (0.6), validation (0.2) and test (0.2) sets
for label in unique_labels:
    labels_path = location + label
    images = os.listdir(labels_path)
    images_df = pd.DataFrame(images)
    train, validate, test = np.split(images_df.sample(frac=1), [int(.6 * len(images_df)), int(.8 * len(images_df))])

    train_file_list = train.values.tolist()

    for filename in train_file_list:
        train_file_paths.append(labels_path + '/' + filename[0])
        train_labels.append(label)

    val_file_list = validate.values.tolist()

    for filename_val in val_file_list:
        val_file_paths.append(labels_path + '/' + filename_val[0])
        val_labels.append(label)

    test_file_list = test.values.tolist()

    for filename_test in test_file_list:
        test_file_paths.append(labels_path + '/' + filename_test[0])
        test_labels.append(label)

txt_paths = ["train_paths.txt",
             "val_paths.txt",
             "test_paths.txt",
             "train_labels.txt",
             "val_labels.txt",
             "test_labels.txt"]
data = [train_file_paths, val_file_paths, test_file_paths, train_labels, val_labels, test_labels]

save_txt_files(txt_paths, data)
