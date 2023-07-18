import csv
import datetime
import re
import os
import uuid

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import easyocr

plt.switch_backend('agg')


app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
reader = easyocr.Reader(['en'], gpu=False)


now = datetime.datetime.now()
timestamp = now.strftime('%Y-%m-%d %H:%M:%S')


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
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
            result = reader.readtext(image_bytes)
            text = result[0][1]
            integers = re.findall(r'\d+', text)
            cleaned_text = ''.join(integers)
            row = {'timestamp': timestamp,
                   'uploaded_image_path': image_path,
                   'cropped_image_path': cropped_image_path,
                   'result_text': cleaned_text}
            fields = ["timestamp", "uploaded_image_path",
                      "cropped_image_path", "result_text"]
            with open('ocr_results.csv', 'a') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writerow(row)
            return render_template("predict.html", result=cropped_image_path, water_meter=cleaned_text)
        else:
            return render_template("predict.html", error="Silahkan upload gambar dengan format JPG")
    else:
        return render_template("predict.html")


@app.route("/history")
def history():
    df = pd.read_csv("ocr_results.csv")
    df = df.sort_values(by=['result_text'])
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

    plot_path = 'static/result/plot.png'
    if not os.path.exists(os.path.dirname(plot_path)):
        os.makedirs(os.path.dirname(plot_path))
    plt.savefig(plot_path)

    return render_template("history.html", df=df, plot_path=plot_path)


@app.route("/trend")
def trend():
    plot_path = 'static/result/plot.png'
    return render_template("trend.html", plot_path=plot_path)


if __name__ == "__main__":
    app.run()
