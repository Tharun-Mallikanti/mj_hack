from flask import Flask, render_template, request, jsonify
from roboflow import Roboflow
import base64
from PIL import Image, ImageDraw
from io import BytesIO
import easyocr
import cv2
app = Flask(__name__)

txtbbs = {}

@app.route('/currency_detection', methods=['GET', 'POST'])
def submit():
    try:
        # Retrieve image data
        image_data_uri = request.json.get('image')

        # Extract base64-encoded part
        _, image_data_base64 = image_data_uri.split(',', 1)

        # Decode base64 image string
        image_bytes = base64.b64decode(image_data_base64)

        # Use BytesIO to create a stream from the image data
        image_stream = BytesIO(image_bytes)

        # Open the image using PIL
        image = Image.open(image_stream).convert('RGB')

        # Save the image to a file
        image.save("input_image.jpg")
        # from roboflow import Roboflow
        # my api key
        rf = Roboflow(api_key="RMzZna7r8BabI0Fz7SJV")
        project = rf.workspace().project("currency_detection-2ukia")
        model = project.version(1).model
        response = model.predict('input_image.jpg').json()
        predicted_currency_note = response['predictions'][0]['predicted_classes'][0]
        
        return jsonify({"result_note":predicted_currency_note})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)