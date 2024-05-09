# Plant Health Detection System

This repository contains a plant health detection system that identifies whether a plant is healthy or suffering from early blight or late blight. It consists of a trained machine learning model, a command-line interface, and a web interface for prediction.

## Folder Structure

- **interface**: Contains the trained machine learning model file (`helplant`) as well as the following Python scripts:
  - **project_plant.py**: Command-line interface script for plant health detection.
  - **web.py**: Web interface script for plant health detection.
- **photos**: Contains the dataset used for the project.
- **templates**: Contains HTML files for the web interface.



## Web Interface

1. Make sure you have Flask installed.
2. Run web.py file 
3. Visit `http://localhost:5000` in your web browser.
4. Upload an image of a plant to get the prediction.

## Sample Usage

For both the command-line interface and web interface, you can use images of plant leaves to get predictions.

## Requirements

- Python 3
- Flask (for the web interface)
- scikit-learn
- Pillow
- numpy





