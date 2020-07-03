import numpy as np
from tensorflow.keras import models
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math
import seaborn as sns
from matplotlib import pyplot as plt

garbage_classification_model = models.load_model('best_model_garbage_xception.hdf5')
# garbage_classification_model = models.load_model('best_model_garbage_VGG16 .hdf5')

size = 299
# size = 224
img_dim = (size, size)
input_shape = (size, size, 3)
batch_size = 32

test_files = []
test_labels = []

data_generator = ImageDataGenerator(rescale=1. / 255)
test_it = data_generator.flow_from_directory(
    directory='test_data',
    target_size=img_dim,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=False,
    seed=42)
var_steps = math.ceil(test_it.n / batch_size)

predictions = garbage_classification_model.predict_generator(test_it, steps=var_steps, workers=0)

predicted_class = np.argmax(predictions, axis=1)
true_class = test_it.classes[test_it.index_array]
class_labels = list(test_it.class_indices.keys())

confusion_matrix_var = (confusion_matrix(
    y_true=true_class,  # ground truth (correct) target values
    y_pred=predicted_class))  # estimated targets as returned by a classifier
print(confusion_matrix_var)

report = classification_report(true_class, predicted_class)
print(report)
categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

sns.heatmap(confusion_matrix_var, annot=True, xticklabels=categories, yticklabels=categories, cmap='Blues', fmt="d")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
