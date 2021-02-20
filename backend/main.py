"""
The flask backend main file
"""
import os
import pickle
import numpy as np
from flask import Flask, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16

app = Flask(__name__)

# This is necessary because QUploader uses an AJAX request
# to send the file
cors = CORS()
cors.init_app(app, resource={r"/api/*": {"origins": "*"}})

model = vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(64, 64, 3))
def apply_vgg(img):
    x_img = np.expand_dims(img, axis=0)
    x_img = vgg16.preprocess_input(x_img)
    # Run the image through the deep neural network to make a prediction
    predictions = model.predict(x_img)

    return predictions

@app.route('/upload', methods=['POST'])
def upload():
    """
    Upload the file from AJAX request
    """
    pred = 0
    for fname in request.files:
        f_uploaded = request.files.get(fname)
        print(f_uploaded)
        f_uploaded.save('backend/uploads/%s' % secure_filename(fname))
        img = image.load_img(f_uploaded, target_size=(224, 224))
        img_array = image.img_to_array(img)
        preds = apply_vgg(img_array)

        kmeans = pickle.load(open("ml/save_tl.pkl", "rb"))
        pred = kmeans.predict(preds)
        topics_data = pd.read_csv('ml/reddit_img_labeled_tl.csv')
        topics_data = topics_data[topics_data['cat'] == pred]
        roast = topics_data['roast'].sample(1)
        roast = roast.to_numpy()
        roast = roast[0]




    return str(roast)

if __name__ == '__main__':
    if not os.path.exists('./uploads'):
        os.mkdir('./uploads')
    app.run()
