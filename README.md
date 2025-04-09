#  American Sign Language (ASL) Detection System

## Objective:-
 Build a system that can detect a given ASL input image and output what the sign represents
 
 
**Real-time ASL alphabet recognition using deep learning and Flask web interface**

- Python
- Flask
- OpenCV
- TensorFlow/Keras

##  Features
- Real-time ASL alphabet detection (A-Z)
- Webcam integration for live prediction
- Simple web interface with Flask
- Responsive design works on desktop/mobile
- Model confidence visualization

## Dataset:-
 Download Link: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

##  Project Structure
```text
flask_app/
├── asldetection.py       # Main Flask app & prediction logic
├── templates/
│   ├── index.html        # Homepage with instructions
│   └── live.html         # Live detection interface
