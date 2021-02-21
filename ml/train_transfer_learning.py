"""
Image recognition and categorization module using
transfer learning and unsupervised learning methods
"""
from io import BytesIO
import pickle
import requests
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16
from sklearn.cluster import MiniBatchKMeans

def input_processing(url):
    """
    Function to download and process the image input
    """
    try:
        response = requests.get(url)
        img = image.load_img(BytesIO(response.content), target_size=(48, 48))
        img_array = image.img_to_array(img)
    except Exception as exception_type:
        print(exception_type)
        img = image.load_img("empty.png", target_size=(48, 48))
        img_array = image.img_to_array(img)

    return img_array

def add_img_array(data):
    """
    Function to add the array to the pandas dataframe
    """
    data['img_array'] = data['url'].map(input_processing)
    data['predictions'] = data['img_array'].apply(apply_vgg)
    data.to_hdf('reddit_img_tl.h5', key='data')

model = vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(48, 48, 3))
def apply_vgg(img):
    """
    Function to add the array to the pandas dataframe
    """
    x_img = np.expand_dims(img, axis=0)
    x_img = vgg16.preprocess_input(x_img)
    # Run the image through the deep neural network to make a prediction
    predictions = model.predict(x_img)

    return predictions

if __name__ == "__main__":
    try:
        print('try')
        topics_data = pd.read_hdf('./reddit_img_tl.h5', key='data')

    except Exception as exception_type:
        print("Error: %s" % exception_type)
        topics_data = pd.read_csv('../redditretriever/reddit.csv')
        add_img_array(topics_data)

    preds = topics_data['predictions']
    x_train = preds.values
    x_train = np.concatenate(x_train)
    x_train = x_train.reshape(len(x_train), -1)

    # Initialize and fit KMeans algorithm
    kmeans = MiniBatchKMeans(n_clusters=100)
    kmeans.fit(x_train)

    # record centroid values
    centroids = kmeans.cluster_centers_

    images = centroids.reshape(100, 1, 1, 512)
    y_pred_kmeans = kmeans.predict(x_train)
    pickle.dump(kmeans, open("save_tl.pkl", "wb"))
    topics_data['cat'] = pd.Series(y_pred_kmeans.astype(int), index=topics_data.index)
    topics_data.to_csv('reddit_img_labeled_tl.csv')
