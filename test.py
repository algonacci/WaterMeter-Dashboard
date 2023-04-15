import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credentials.json'
from google.auth import credentials

from google.cloud import vision

client = vision.ImageAnnotatorClient(credentials="credentials.json")
response = client.annotate_image({
  'image': {'source': {'image_uri': 'https://algonacci.github.io/img/Profile_%20(7).jpg'}},
  'features': [{'type_': vision.Feature.Type.FACE_DETECTION}]
})