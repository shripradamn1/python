# %%
import splitfolders
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

import os
import plotly.express as px
import numpy as np
# import splitfolders
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns

# %%
class_names = ['Cyst', 'Normal', 'Tumor', 'Stone'] 

cyst = len(os.listdir('Cyst'))
normal = len(os.listdir('Normal'))
stone = len(os.listdir('Stone'))
tumor = len(os.listdir('Tumor'))

images = [cyst, normal, stone, tumor]

plt.figure(figsize=(6,5))
sns.barplot(x= class_names, y= images, palette= sns.cubehelix_palette())
plt.title('Frequency of Each Class in the Dataset', fontsize=14)
plt.xlabel('Class names', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(range(len(images)))
plt.show()
px.pie(names= class_names, values= images, color_discrete_sequence=px.colors.sequential.BuGn)

# %%
splitfolders.ratio(
    "../CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
   output="./dataset",
   ratio=(.8,.1,.1)
)

# %%
train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)
val = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory('./dataset/train',
                                          target_size=(180, 180),
                                          color_mode='grayscale',
                                          class_mode = 'categorical',
                                          batch_size=64
                                         )

test_dataset = test.flow_from_directory('./dataset/test',
                                        target_size=(180, 180),
                                        color_mode='grayscale',
                                        class_mode = 'categorical',                                  
                                        batch_size=64,
                                        shuffle = False
                                       )

valid_dataset = val.flow_from_directory('./dataset/val',
                                        target_size=(180, 180),
                                        color_mode='grayscale',
                                        class_mode = 'categorical',
                                        batch_size=64
                                       )

# %%
type(train_dataset)

# %%
class_names = ['Cyst', 'Normal', 'Tumor', 'Stone'] 
def class_type(dataset, n_images):

    i = 1
    images, labels = next(dataset)
    labels = labels.astype('int32')

    plt.figure(figsize=(14, 15))
    
    for image, label in zip(images, labels):
        plt.subplot(4, 3, i)
        plt.imshow(image)
        plt.title(class_names[np.argmax(label)])
        plt.axis('off')
        i += 1
        if i == n_images:
            break
    plt.show()

class_type(train_dataset, 10)

# %%
inputs = keras.Input(shape=(180,180,1))
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(inputs)
x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(512,activation='relu')(x)
outputs = layers.Dense(4, activation='softmax')(x)
model = keras.Model(inputs = inputs, outputs=outputs)
model.summary()

# %%
model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy', 
                                                                                   keras.metrics.Precision(name='precision'),
                                                                                   keras.metrics.Recall(name='recall')])
history = model.fit(train_dataset,
          validation_data=valid_dataset,
          epochs=3
         )

# %%
fig, ax = plt.subplots(1, 4, figsize=(20, 3))
ax = ax.ravel()

for i, met in enumerate(['precision', 'recall', 'accuracy', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])

# %%
predictions = model.predict(test_dataset)
model.evaluate(test_dataset)

# %%
def class_prediction(dataset, n_images):
    i = 1
    images, labels = next(dataset)

    preds = model.predict(images)
    predictions = np.argmax(preds, axis=1)
    labels = np.argmax(labels, axis= 1)
    plt.figure(figsize=(14, 15))
    for image, label in zip(images, labels):
        plt.subplot(4, 3, i)
        plt.imshow(image)
        if predictions[i] == labels[i]:
            title_obj = plt.title(class_names[label])
            plt.setp(title_obj, color='g') 
            plt.axis('off')
        else:
            title_obj = plt.title(class_names[label])
            plt.setp(title_obj, color='r') 
            plt.axis('off')
        i += 1
        if i == n_images:
            break
    plt.show()

class_prediction(test_dataset, 10)


# %%
diseases_labels = []

for key, value in train_dataset.class_indices.items():
   diseases_labels.append(key)

def evaluate(actual, predictions):
  pre = []
  for i in predictions:
    pre.append(np.argmax(i))

  accuracy = (pre == actual).sum() / actual.shape[0]
  print(f'Accuracy: {accuracy}')
  precision, recall, f1_score, _ = precision_recall_fscore_support(actual, pre, average='macro')
  print(f'Precision: {precision}')
  print(f'Recall: {recall}')
  print(f'F1_score: {f1_score}')

  fig, ax = plt.subplots(figsize=(20,20))
  conf_mat = confusion_matrix(actual, pre)
  sns.heatmap(conf_mat, annot=True, fmt='.0f', cmap="YlGnBu", xticklabels=diseases_labels, yticklabels=diseases_labels).set_title('Confusion Matrix Heat map')
  plt.show()
    
evaluate(test_dataset.classes, predictions)

# %%



