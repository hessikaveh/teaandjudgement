"""
The flask backend main file
"""
import os
import pickle
from flask import Flask, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
from PIL import Image
from keras.preprocessing.image import img_to_array

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
        img_array = img_to_array(img_array)
        img_array = img_array/255.
        img_array = img_array.reshape(len(img_array), -1)
        kmeans = pickle.load(open("../ml/save.pkl", "rb"))
        pred = kmeans.predict(img_array)
        pred = int(pred.mean())


    return str(pred)

if __name__ == '__main__':
    if not os.path.exists('./uploads'):
        os.mkdir('./uploads')
    app.run(debug=True)
