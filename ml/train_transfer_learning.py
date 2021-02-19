"""
Image recognition and categorization module using
transfer learning and unsupervised learning methods
"""
from io import BytesIO
import requests
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import pickle

def input_processing(url):
    """
    Function to download and process the image input
    """
    try:
        response = requests.get(url)
        img = image.load_img(BytesIO(response.content), target_size=(224, 224))
        img_array = image.img_to_array(img)
        #plt.imshow(img)
        #plt.show()
    except Exception as exception_type:
        print(exception_type)
        img = image.load_img("empty.png", target_size=(224, 224))
        img_array = image.img_to_array(img)

    return img_array

def add_img_array(data):
    """
    Function to add the array to the pandas dataframe
    """
    data['img_array'] = data['url'].map(input_processing)
    data['predictions'] = data['img_array'].apply(apply_vgg)
    data.to_hdf('reddit_img_tl.h5', key='data')

model = vgg16.VGG16()
def apply_vgg(img):
    x = np.expand_dims(img, axis=0)
    x = vgg16.preprocess_input(x)
    # Run the image through the deep neural network to make a prediction
    predictions = model.predict(x)

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
    print(x_train[0])
    print(x_train[0].shape)
    print(x_train.shape)
    x_train = np.concatenate(x_train)
    #print(x_train)
    #print(len(x_train))
    #print(np.prod(x_train.shape[1:]))
    x_train = x_train.reshape(len(x_train), -1)

    #print(x_train.shape)


    # Initialize and fit KMeans algorithm
    kmeans = MiniBatchKMeans(n_clusters=100)
    kmeans.fit(x_train)

    # record centroid values
    centroids = kmeans.cluster_centers_
    print(centroids)
    images = centroids.reshape(100, 1, 1000)
    for predictions in images:
        # Look up the names of the predicted classes. Index zero is the results for the first image.
        predicted_classes = vgg16.decode_predictions(predictions)

        print("Top predictions for this image:")

        for imagenet_id, name, likelihood in predicted_classes[0]:
            print("Prediction: {} - {:2f}".format(name, likelihood))
    #kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
    y_pred_kmeans = kmeans.predict(x_train)
    print(y_pred_kmeans)
    print(y_pred_kmeans.shape)

    pickle.dump(kmeans, open("save_tl.pkl", "wb"))
    topics_data['cat'] = pd.Series(y_pred_kmeans.astype(int), index=topics_data.index)
    print(topics_data)
    topics_data.to_csv('reddit_img_labeled_tl.csv')

