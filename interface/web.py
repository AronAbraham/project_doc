 import numpy as np
from flask import Flask, request, render_template
import pickle
from PIL import Image
import io

# Create an app object using the Flask class.
app = Flask(__name__)

# Load the plant health detection model (Pickle file)
model = pickle.load(open('helplant', 'rb'))

# Define the route for the home page.
@app.route('/')
def home():
    return render_template('index.html')

# Add POST method to the decorator to allow for image upload.
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        uploaded_image = request.files['image']

        if uploaded_image:
            # Process the image
            image = Image.open(uploaded_image)
            image = image.resize((256, 256))  # Resize the image to the expected input size
            image_array = np.array(image)  # Convert the image to a numpy array
            image_array = image_array / 255.0  # Normalize the image data if needed

            # Make a prediction using the loaded model
            prediction = model.predict(np.expand_dims(image_array, axis=0))

            # Determine the class label based on the maximum prediction value
            max_index = np.argmax(prediction)
            class_labels = ['Early Blight','Healthy Plant','Late Blight']
            predicted_label = class_labels[max_index]

            # You can do something with the predicted_label here
            return render_template('result.html', predicted_label=predicted_label)
        else:
            # Handle the case where no image was uploaded
            return "No image uploaded."

if __name__ == '__main__':
    app.run(debug=True)
