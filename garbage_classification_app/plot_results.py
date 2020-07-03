import pandas as pd
from matplotlib import pyplot as plt

location = 'logs/csv_results/'

train_vgg_acc = pd.read_csv(location + 'run-scalars_VGG16-tag-epoch_acc.csv')
train_xception_acc = pd.read_csv(location + 'run-scalars_Xception-tag-epoch_acc.csv')

plt.style.use('ggplot')
for idx, frame in enumerate([train_vgg_acc, train_xception_acc]):
    plt.plot(frame['Step'], frame['Value'])

plt.legend(['VGG16', 'Xception'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training accuracy')
plt.show()

val_vgg_acc = pd.read_csv(location + 'run-scalars_VGG16-tag-epoch_val_acc.csv')
val_xception_acc = pd.read_csv(location + 'run-scalars_Xception-tag-epoch_val_acc.csv')

plt.style.use('ggplot')
for idx, frame in enumerate([val_vgg_acc, val_xception_acc]):
    plt.plot(frame['Step'], frame['Value'])

plt.legend(['VGG16', 'Xception'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation accuracy')
plt.show()
