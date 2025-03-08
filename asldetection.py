# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.preprocessing import image
import cv2  

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths to the dataset
train_dir = r'add your training dataset path'

# Image dimensions and batch size
img_width, img_height = 64, 64
batch_size = 32

# Define the class labels (A-Z, SPACE, DELETE, NOTHING)
class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'SPACE', 'DELETE', 'NOTHING'
]

# Function to train and save the model
def train_and_save_model():
    # Data preprocessing and augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # Split 20% of the data for validation
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'  # Use this for training data
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'  # Use this for validation data
    )

    # Build the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(29, activation='softmax')  # 29 classes for ASL letters and symbols
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    print("Training the model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )

    # Save the model
    model.save('asl_detection_model.h5')
    print("Model saved as 'asl_detection_model.h5'")

# Initialize Flask app
app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route for uploaded images
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(64, 64))  # Resize to match model input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image
        
        # Make a prediction
        model = load_model('asl_detection_model.h5')
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_label = class_labels[predicted_class[0]]
        
        return render_template('index.html', prediction=predicted_label, image_path=file_path)

# Live camera prediction route
@app.route('/live')
def live():
    return render_template('live.html')

# Function to process live camera input
def process_live_camera():
    # Load the model
    model = load_model('asl_detection_model.h5')
    
    # Open the camera
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        resized_frame = cv2.resize(frame, (img_width, img_height))
        normalized_frame = resized_frame.astype('float32') / 255.0
        input_frame = np.expand_dims(normalized_frame, axis=0)
        
        # Make a prediction
        prediction = model.predict(input_frame)
        predicted_class = np.argmax(prediction, axis=1)
        predicted_label = class_labels[predicted_class[0]]
        
        # Display the prediction on the frame
        cv2.putText(frame, f'Prediction: {predicted_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Live ASL Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

# Run the Flask app
if __name__ == '__main__':
    # Train and save the model if it doesn't exist
    if not os.path.exists('asl_detection_model.h5'):
        train_and_save_model()
    
    # Start the Flask app
    app.run(debug=True)
