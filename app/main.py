import os
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
import soundfile as sf
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.core.audio import SoundLoader

# Define DATA_PATH (modify as per your dataset location)
DATA_PATH = r'C:\Users\PRAVEEN ANANTH\OneDrive\Desktop\speech_emotion_recognition\data\RAVDESS'

# Check if dataset path exists
if os.path.exists(DATA_PATH):
    print(f"Dataset path exists: {DATA_PATH}")
else:
    raise ValueError("Dataset path does not exist. Please check the DATA_PATH.")

# Emotion labels based on RAVDESS documentation
emotion_labels = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Function to extract MFCC features
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}, {e}")
        return None

# Train the model if not already done
def train_model():
    features = []
    emotions = []

    # Traverse dataset and extract features
    for actor in os.listdir(DATA_PATH):
        actor_path = os.path.join(DATA_PATH, actor)
        if not os.path.isdir(actor_path):
            print(f"Skipping {actor_path}, not a directory.")
            continue
        print(f"Processing actor directory: {actor_path}")
        for file in os.listdir(actor_path):
            if file.endswith('.wav'):
                file_path = os.path.join(actor_path, file)
                emotion_code = file.split('-')[2]
                emotion = emotion_labels.get(emotion_code, 'unknown')
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    emotions.append(emotion)
                else:
                    print(f"Failed to extract features from: {file_path}")

    if not features or not emotions:
        raise ValueError("No features or emotions extracted from the dataset. Please check your data source.")

    # Create DataFrame and encode emotions
    df = pd.DataFrame(features)
    df['emotion'] = emotions
    le = LabelEncoder()
    y = le.fit_transform(df['emotion'])
    joblib.dump(le, 'model/label_encoder.pkl')

    # Scale features
    X = df.drop('emotion', axis=1).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'model/scaler.pkl')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    # Model definition
    model = Sequential()
    model.add(Dense(256, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(le.classes_), activation='softmax'))

    # Compile and train model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train_cat, epochs=50, batch_size=32, validation_data=(X_test, y_test_cat), verbose=1)
    model.save('model/emotion_recognition_model.h5')

    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('model/emotion_recognition_model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("Model training and conversion completed!")

# Kivy App for real-time prediction and audio processing
class SERApp(App):
    def build(self):
        # Train model if not done already
        if not os.path.exists('model/emotion_recognition_model.tflite'):
            print("Training model...")
            train_model()

        # Create layout
        layout = BoxLayout(orientation='vertical')
        self.result_label = Label(text="Prediction Result will appear here", size_hint=(1, 0.2))
        layout.add_widget(self.result_label)

        # Record and upload buttons
        record_button = Button(text="Record Audio", size_hint=(1, 0.2))
        record_button.bind(on_press=self.record_audio)

        upload_button = Button(text="Upload Audio", size_hint=(1, 0.2))
        upload_button.bind(on_press=self.upload_audio)

        layout.add_widget(record_button)
        layout.add_widget(upload_button)

        return layout

    def upload_audio(self, instance):
        print("Opening file chooser...")  # Log message
        file_chooser = FileChooserIconView(filters=['*.wav'])  # Filter to .wav files only
        file_chooser.bind(on_selection = self.selected_file)  # Bind the file selection handler

        # Create and open a popup for file chooser
        self.popup = Popup(
            title="Select a .wav file",
            content=file_chooser,
            size_hint=(0.9, 0.9)
        )
        self.popup.bind(on_dismiss=self.on_popup_dismiss)  # Add dismissal handling
        self.popup.open()

    def on_popup_dismiss(self, instance):
        print("Popup dismissed.")  # This will confirm if popup closes properly

    def selected_file(self, instance, selection):
        print("on_selection event triggered.")
        print(f"Selection received: {selection}")  # Log message

        if selection:  # Check if a file is selected
            selected_path = os.path.normpath(selection[0])  # Get the first selected file
            print(f"Selected file: {selected_path}")  # Log the selected file path

            if selected_path.endswith('.wav'):
                self.popup.dismiss()  # Close the popup
                print("Valid .wav file selected. Starting prediction...")
                self.predict_emotion(selected_path)
            else:
                print("Error: Selected file is not a .wav file.")
                self.result_label.text = "Please select a valid .wav file."
        else:
            print("No file selected.")
            self.result_label.text = "No file selected."

    def predict_emotion(self, selected_path):
        print(f"Starting emotion prediction for file: {selected_path}")
        try:
            # Step 1: Extract features
            features = extract_features(selected_path)
            if features is None:
                raise ValueError("Failed to extract features.")
            print("Features successfully extracted.")

            # Step 2: Load Scaler and Label Encoder
            try:
                scaler = joblib.load('model/scaler.pkl')
                le = joblib.load('model/label_encoder.pkl')
                print("Scaler and Label Encoder loaded successfully.")
            except Exception as e:
                print(f"Error loading models: {e}")
                self.result_label.text = "Error loading models."
                return

            # Step 3: Scale Features
            features_scaled = scaler.transform([features])
            print("Features scaled successfully.")

            # Step 4: Load TFLite Model
            try:
                interpreter = tf.lite.Interpreter(model_path='model/emotion_recognition_model.tflite')
                interpreter.allocate_tensors()
                print("TFLite model loaded and tensors allocated.")
            except Exception as e:
                print(f"Error loading TFLite model: {e}")
                self.result_label.text = "Error loading TFLite model."
                return

            # Step 5: Perform Prediction
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_data = np.array(features_scaled, dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = np.argmax(output_data)
            emotion = le.inverse_transform([prediction])[0]

            print(f"Predicted emotion: {emotion}")
            self.result_label.text = f"Prediction Result: {emotion}"

            # Step 6: Play the Audio File
            sound = SoundLoader.load(selected_path)
            if sound:
                sound.play()
                print("Playing audio file.")
            else:
                print("Error: Failed to load audio for playback.")
        except Exception as e:
            print(f"Error during prediction: {e}")
            self.result_label.text = "Error during prediction."

    def record_audio(self, instance):
        print("Recording audio...")

        # Recording settings
        duration = 5  # in seconds
        fs = 16000  # Sample rate
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()

        # Save the recorded file
        recorded_file = 'recorded_audio.wav'
        sf.write(recorded_file, audio_data, fs)
        print(f"Audio recorded and saved to {recorded_file}")

        # Predict emotion of recorded audio
        self.predict_emotion(recorded_file)

if __name__ == "__main__":
    SERApp().run()
