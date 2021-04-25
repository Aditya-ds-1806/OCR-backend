from flask import Flask, request
from flask_cors import CORS
from ocr import recognizeText
import json
import os

if(not(os.path.exists('./output_imgs'))):
    os.mkdir('./output_imgs')

app = Flask(__name__)
CORS(app, resources=r'/')


@app.route('/', methods=['POST'])
def recognize():
    opts = {
        'alignment': bool(int(request.form['alignment'])),
        'gaussian': bool(int(request.form['gaussian'])),
        'ed': bool(int(request.form['ed'])),
        'median': bool(int(request.form['median']))
    }
    img = request.files['img']
    img.save('./output_imgs/image.png')
    text = recognizeText('./output_imgs/image.png', opts)
    return json.dumps({'text': text})
