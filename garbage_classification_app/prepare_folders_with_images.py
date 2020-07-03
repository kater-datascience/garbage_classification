import os
import json
import shutil


def copy_files(file_paths, output_directories):
    for idx, file in enumerate(file_paths):
        with open(file, 'r') as file_handle:
            dataset_filepath = json.load(file_handle)
        os.mkdir(output_directories[idx]) if not os.path.isdir(output_directories[idx]) else None
        for fn in dataset_filepath:
            class_name = fn.split('/')[3]
            dir_custom = os.path.join(output_directories[idx], class_name)
            os.mkdir(dir_custom) if not os.path.isdir(dir_custom) else None
            shutil.copy(fn, dir_custom)


txt_paths = ["labels_txt/train_paths.txt",
             "labels_txt/val_paths.txt",
             "labels_txt/test_paths.txt"]
dirs = ["training_data",
        "validation_data",
        "test_data"]

copy_files(txt_paths, dirs)
