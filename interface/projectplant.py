from sklearn.externals import joblib
from PIL import Image
import numpy as np

# Load the model
loaded_model = joblib.load('helpplant')

# Open and load the image using Pillow
image = Image.open('healthy.jpg')

# Check if the image was loaded successfully
if image is not None:
    # Resize the image to the desired size (e.g., 224x224)
    image = image.resize((256, 256))

    # Convert the image to float32
    image = np.array(image, dtype=np.float32)

    # Normalize pixel values to the range [0, 1]
    image /= 255.0

    # Make predictions using the loaded model
    predictions = loaded_model.predict(np.expand_dims(image, axis=0))

    class_labels = ['early blight', 'late blight', 'healthy']

# Find the index of the maximum value in the predictions vector
    predicted_class_index = np.argmax(predictions)

# Get the corresponding class label
    predicted_class = class_labels[predicted_class_index]

# Print the predicted class
    print("Predicted Class:", predicted_class)
else:
    print("Failed to load the image.")
