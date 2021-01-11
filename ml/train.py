"""
Unsupervised image classification module
"""
import pickle
from io import BytesIO
import pandas as pd
import numpy as np
from PIL import Image
import requests
from keras.preprocessing.image import img_to_array
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

def input_processing(url):
    """
    Function to download and process the image input
    """
    try:
        response = requests.get(url)
        img_array = (Image.open(BytesIO(response.content)).convert('L')).resize((400, 400))
        img_array = img_to_array(img_array)
    except Exception as exception_type:
        print(exception_type)
        empty_img = Image.new('L', (400, 400))
        img_array = empty_img.resize((400, 400))
        img_array = img_to_array(img_array)

    return img_array


def add_img_array(data):
    """
    Function to add the array to the pandas dataframe
    """
    data['img_array'] = data['url'].map(input_processing)
    data.to_hdf('reddit_img.h5', key='data')

if __name__ == "__main__":
    try:
        print('try')
        topics_data = pd.read_hdf('./reddit_img.h5', key='data')

    except Exception as exception_type:
        print("Error: %s" % exception_type)
        topics_data = pd.read_csv('../redditretriever/reddit.csv')
        add_img_array(topics_data)


    x_train = topics_data['img_array']
    print(x_train[3])
    x_train = x_train / 255.
    x_train = x_train.values
    print(x_train[0].shape)
    print(x_train[1].shape)
    print(x_train[2].shape)
    print(x_train[3].shape)

    x_train = np.concatenate(x_train)
    print(len(x_train))
    print(np.prod(x_train.shape[1:]))
    x_train = x_train.reshape(len(x_train), -1)

    print(x_train.shape)


    # Initialize and fit KMeans algorithm
    kmeans = MiniBatchKMeans(n_clusters=100)
    kmeans.fit(x_train)

    # record centroid values
    centroids = kmeans.cluster_centers_
    print(centroids)
    # reshape centroids into images
    images = centroids.reshape(100, 20, 20)
    images *= 255
    images = images.astype(np.uint8)
    print(images)

    # determine cluster labels

    # create figure with subplots using matplotlib.pyplot
    fig, axs = plt.subplots(10, 10, figsize=(20, 20))
    plt.gray()

    # loop through subplots and add centroid images
    for i, ax in enumerate(axs.flat):
        # add image to subplot
        ax.matshow(images[i])
        ax.axis('off')

    # display the figure
    #fig.show()
    plt.savefig('foo.png')

    # reshape centroids into images
    images = x_train.reshape((961, 400, 400))
    images *= 255
    images = images.astype(np.uint8)

    # create figure with subplots using matplotlib.pyplot
    fig, axs = plt.subplots(30, 30, figsize=(20, 20))
    plt.gray()

    # loop through subplots and add centroid images
    for i, ax in enumerate(axs.flat):
        # add image to subplot
        ax.matshow(images[i])
        ax.axis('off')

    # display the figure
    #fig.show()
    plt.savefig('fooInput.png')

    #kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
    y_pred_kmeans = kmeans.predict(x_train)
    print(y_pred_kmeans)
    print(y_pred_kmeans.shape)
    final_image = y_pred_kmeans.reshape(961, 20, 20)
    means = final_image.mean(axis=1)
    print(final_image)
    print(kmeans.labels_)
    means = means.mean(axis=1)
    print(means)
    pickle.dump(kmeans, open("save.pkl", "wb"))
    topics_data['cat'] = pd.Series(means.astype(int), index=topics_data.index)
    print(topics_data)
    topics_data.to_csv('reddit_img_labeled.csv')
