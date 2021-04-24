from flask import Flask, request
from ocr import recognizeText
import json
from os import mkdir
import os.path as path

if(not(path.exists('./output_imgs'))):
    mkdir('./output_imgs')

app = Flask(__name__)


@app.route('/', methods=['POST'])
def hello_world():
    alignment = bool(int(request.form['alignment']))
    print(alignment)
    img = request.files['img']
    img.save('./output_imgs/image.png')
    text = recognizeText('./output_imgs/image.png', alignment)
    return json.dumps({'text': text})
