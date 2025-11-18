# 🎤 Speech Emotion Recognition (SER)

A machine learning project that detects human emotions from speech using **audio preprocessing** and **deep learning (TensorFlow)**.
The system records or loads an audio file, extracts features using **Librosa**, and predicts the emotion using a trained model.
A simple **Kivy UI** is included to interactively record, process, and display the predicted emotion.

---

## 📌 Features

* 🎙 **Record live audio** using `sounddevice`
* 🎵 **Extract MFCCs and audio features** using `librosa`
* 🧠 **Deep learning model (TensorFlow) for emotion classification**
* 📊 Data processing, visualization, and training notebooks
* 🖥 **Kivy-based UI** for user interaction
* 💾 Save & load trained models using `joblib` or `.h5`
* 🔊 Works on WAV audio files

---

## 📁 Project Structure

Matches your actual folder structure:

```
speech-emotion-recognition/
│
├── app/                  # Kivy app code/UI
├── data/                 # Audio dataset
├── model/                # Saved trained models (.h5, .pkl)
├── notebooks/            # Jupyter notebooks for training & experiments
├── venv/                 # Virtual environment (ignored in git)
│
├── recorded_audio.wav    # Example recorded audio
├── temp.wav              # Temporary recording file
├── test_kivy.py          # Kivy test script
├── summa.ipynb           # Notebook for experiments
├── .gitignore
└── README.md
```

---

## 🛠 Technologies Used

* **Kivy** – User Interface
* **TensorFlow / Keras** – Deep learning model
* **Librosa** – Audio feature extraction
* **NumPy / Pandas** – Data handling
* **scikit-learn** – Feature scaling + classical models
* **joblib** – Saving ML models
* **sounddevice** – Recording audio
* **matplotlib / seaborn** – Visualization

---

## 🎙 How It Works

1. User records audio or selects a WAV file
2. Audio is processed using Librosa
3. MFCCs + other features are extracted
4. Features are fed into the trained TensorFlow model
5. The model outputs a predicted emotion (e.g., Happy, Angry, Neutral)
6. Kivy app displays the result

---

## 📊 Model

The model is a deep learning classifier trained on MFCC features.
Typical architecture:

* Dense Layers
* Dropout
* Softmax output (multi-class emotion prediction)

---

## 😊 Supported Emotions

(Depends on your dataset, example:)

* Happy
* Sad
* Angry
* Neutral
* Fear
* Surprise

---

## 🔮 Future Improvements

* Add mobile APK using **Kivy + Buildozer**
* Use CNN or LSTM models for better accuracy
* Add real-time continuous emotion tracking
* Deploy as a web app

---
