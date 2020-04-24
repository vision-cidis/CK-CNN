from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image
from keras.models import load_model
from sklearn.metrics import plot_confusion_matrix

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import keras,os

#INDIVIDUAL PREDICTION
# img = image.load_img("sample4.png",target_size=(224,224))
# img = np.asarray(img)
# plt.imshow(img)
# img = np.expand_dims(img, axis=0)
# from keras.models import load_model
# saved_model = load_model("vgg16_4.h5")
# output = saved_model.predict(img)
# print("garbage",output[0][0])
# print("good",output[0][1])
# print("broken",output[0][2])
# print("rotten",output[0][3])


def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 

#from keras.models import load_weights
saved_model = load_model("weights3class.h5")
saved_model.summary()



test_generator = ImageDataGenerator()
test_data_generator = test_generator.flow_from_directory(
    directory="dataset3class/validation_set/validation_set", # Put your path here
    target_size=(224,224),
    shuffle=False)
test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)

predictions = saved_model.predict_generator(test_data_generator, steps=test_steps_per_epoch)
# Get most likely class
predicted_classes = np.argmax(predictions, axis=1) #ytest

###########################################
true_classes = test_data_generator.classes
class_labels = list(test_data_generator.class_indices.keys())   


report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
cm = metrics.confusion_matrix(true_classes, predicted_classes,labels=[0,1], normalize='true')

print(cm)

print(report)    

print("Accuracy: ", accuracy(cm))


# plt.show()
fig = plt.figure() 
ax = fig.add_subplot(111) 
cax = ax.matshow(cm) 
plt.title('Confusion matrix of the classifier') 
fig.colorbar(cax) 
ax.set_xticklabels([''] + class_labels) 
ax.set_yticklabels([''] + class_labels) 
plt.xlabel('Predicted') 
plt.ylabel('True')
plt.show()



from sklearn.preprocessing import label_binarize
n_classes = 2

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

# Plot linewidth.
lw = 3

# Binarize the output
y = label_binarize(true_classes, classes=[0, 1])
n_classes = y.shape[1]
print("Y")
print(y)
# Compute ROC curve and ROC area for each class
print(y.shape)
print(predictions.shape)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    print("param1 ")
    print(y)
    print("param2 ")
    print( predictions[:, i])
    fpr[i], tpr[i], _ = roc_curve(y, predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print("fpr")
    print(fpr[i])
    print("tpr")
    print( tpr[i])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(2)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
mean_tpr /= 2

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(1,2), colors):
    plt.plot(fpr[i], tpr[i], '-',color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

#'-.' CKCNN
#':' VGG
#'--' RESNET

#plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

