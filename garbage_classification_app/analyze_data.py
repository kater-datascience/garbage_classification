import os
import seaborn as sns
import matplotlib.pyplot as plt

# declare variables
location = '../Garbage classification/Garbage classification/'
model_labels = []

for path, subdirs, files in os.walk(location):
    for name in files:
        model_labels.append(os.path.split(path)[1])

sns.set(style="darkgrid")
ax = sns.countplot(x=model_labels,  palette="Set3")
ax.set(xlabel='garbage type', ylabel='images count')
plt.show()
