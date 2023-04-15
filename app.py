from flask import Flask, request, jsonify
from google.cloud import vision

app = Flask(__name__)


@app.route("/")
def index():
    return "Hello World!"

@app.route('/detect-ocr', methods=['POST'])
def detect_ocr():
    # Get the image file from the request
    image_file = request.files.get('image')
    
    # Create a client for the Google Cloud Vision API
    client = vision.ImageAnnotatorClient()
    
    # Read the contents of the image file
    image_content = image_file.read()
    
    # Create an image object
    image = vision.Image(content=image_content)
    
    # Configure the OCR feature
    ocr_feature = vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION)
    
    # Perform text detection on the image
    response = client.annotate_image({
        'image': image,
        'features': [ocr_feature],
    })
    
    # Get the text annotations from the response
    annotations = response.text_annotations
    
    # Return the first annotation (which should contain the entire text)
    if annotations:
        text = annotations[0].description
        return jsonify({'text': text})
    
    # If no text annotations were found, return an error message
    else:
        return jsonify({'error': 'No text found in the image'})


if __name__ == "__main__":
    app.run()
