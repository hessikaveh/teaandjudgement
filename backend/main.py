"""
The flask backend main file
"""
import os
import pickle
import numpy as np
from flask import Flask, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
from PIL import Image
import pandas as pd

app = Flask(__name__)

# This is necessary because QUploader uses an AJAX request
# to send the file
cors = CORS()
cors.init_app(app, resource={r"/api/*": {"origins": "*"}})

@app.route('/upload', methods=['POST'])
def upload():
    """
    Upload the file from AJAX request
    """
    pred = 0
    for fname in request.files:
        f_uploaded = request.files.get(fname)
        print(f_uploaded)
        f_uploaded.save('./uploads/%s' % secure_filename(fname))

        img_array = (Image.open(f_uploaded).convert('L')).resize((400, 400))
        img_array = np.array(img_array)
        img_array = img_array/255.
        img_array = img_array.reshape(len(img_array), -1)
        kmeans = pickle.load(open("ml/save.pkl", "rb"))
        pred = kmeans.predict(img_array)
        pred = int(pred.mean())
        topics_data = pd.read_csv('ml/reddit_img_labeled.csv')
        topics_data = topics_data[topics_data['cat'] == pred]
        roast = topics_data['roast'].sample(1)
        roast = roast.to_numpy()
        roast = roast[0]




    return str(roast)

if __name__ == '__main__':
    if not os.path.exists('./uploads'):
        os.mkdir('./uploads')
    app.run()
