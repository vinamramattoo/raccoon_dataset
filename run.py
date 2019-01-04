import numpy as np
import pandas as pd

np.random.seed(1)

full_labels = pd.read_csv('data/raccoon_labels.csv')
grouped = full_labels.groupby('filename')
grouped.apply(lambda x: len(x)).value_counts()

gb = full_labels.groupby('filename')
grouped_list = [gb.get_group(x) for x in gb.groups]
len(grouped_list)
train_index = np.random.choice(len(grouped_list), size=40, replace=False)
test_index = np.setdiff1d(list(range(50)), train_index)
len(train_index), len(test_index)


# take first 200 files
train = pd.concat([grouped_list[i] for i in train_index])
test = pd.concat([grouped_list[i] for i in test_index])

len(train), len(test)

train.to_csv('train_labels.csv', index=None)
test.to_csv('test_labels.csv', index=None)

import cv2
import pandas as pd
from PIL import Image

full_labels = pd.read_csv('data/raccoon_labels.csv')

def draw_boxes(image_name):
    selected_value = full_labels[full_labels.filename == image_name]
    img = cv2.imread('images/{}'.format(image_name))
    for index, row in selected_value.iterrows():
        img = cv2.rectangle(img, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (0, 255, 0), 3)
    return img


img = Image.fromarray(draw_boxes('raccoon-100.jpg'))

img.show()