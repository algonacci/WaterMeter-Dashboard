import csv
import datetime
import os
import uuid

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from flask import Flask, render_template, request
from google.cloud import vision
from google.oauth2.service_account import Credentials
from werkzeug.utils import secure_filename

plt.switch_backend('agg')


app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
creds = Credentials.from_service_account_file('credentials2.json')
client = vision.ImageAnnotatorClient(credentials=creds)


now = datetime.datetime.now()
timestamp = now.strftime('%Y-%m-%d %H:%M:%S')


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            image.save(os.path.join(
                app.config['UPLOAD_FOLDER'], secure_filename(image.filename)))
            image_path = os.path.join(
                app.config['UPLOAD_FOLDER'], secure_filename(image.filename))
            model = torch.hub.load("ultralytics/yolov5",
                                   "custom",
                                   path="./best.pt",
                                   force_reload=True)
            result = model(image_path)
            print(result)
            bbox_raw = result.xyxy[0][0]
            bbox = []
            for bound in bbox_raw:
                bbox.append(int(bound.item()))
            bbox = bbox[:4]
            image = cv2.imread(image_path)
            cropped_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            dynamic_filename = str(uuid.uuid4())
            cv2.imwrite("static/result/cropped_image_" +
                        dynamic_filename + ".jpg", cropped_image)
            cropped_image_path = "static/result/cropped_image_" + dynamic_filename + ".jpg"
            with open(cropped_image_path, 'rb') as f:
                image_bytes = f.read()
            image = vision.Image(content=image_bytes)
            response = client.text_detection(image=image)
            text = response.text_annotations[0].description
            row = {'timestamp': timestamp,
                   'uploaded_image_path': image_path,
                   'cropped_image_path': cropped_image_path,
                   'result_text': text}
            fields = ["timestamp", "uploaded_image_path",
                      "cropped_image_path", "result_text"]
            with open('ocr_results.csv', 'a') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writerow(row)
            return render_template("index.html", result=cropped_image_path, water_meter=text)
        else:
            return render_template("index.html", error="Silahkan upload gambar dengan format JPG")
    else:
        return render_template("index.html")


@app.route("/history")
def history():
    df = pd.read_csv("ocr_results.csv")
    df = df.sort_values(by=['timestamp'])
    x = list(range(len(df)))
    data = [int(val) for val in df['result_text'].iloc[:]]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, data, marker='o', markersize=8,
            linestyle='-', linewidth=1, label='Result Text')
    ax.set_xlabel('Data Point')
    ax.set_ylabel('Result Text')
    ax.set_title('Result Text Trend')
    for i, val in enumerate(data):
        ax.text(x[i], data[i], str(val), ha='center', va='bottom')

    # save the plot to a file
    plot_path = 'static/result/plot.png'
    if not os.path.exists(os.path.dirname(plot_path)):
        os.makedirs(os.path.dirname(plot_path))
    plt.savefig(plot_path)

    return render_template("history.html", df=df, plot_path=plot_path)


if __name__ == "__main__":
    app.run()
