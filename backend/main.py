"""
The flask backend main file
"""
import os
from flask import Flask, request
from werkzeug import secure_filename
from flask_cors import CORS

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
    for fname in request.files:
        f_uploaded = request.files.get(fname)
        print(f_uploaded)
        app.logger.info('Helloa!')
        f_uploaded.save('./uploads/%s' % secure_filename(fname))

    return 'Okay!'

if __name__ == '__main__':
    if not os.path.exists('./uploads'):
        os.mkdir('./uploads')
    app.run(debug=True)
