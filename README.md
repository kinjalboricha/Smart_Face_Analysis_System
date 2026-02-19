# Smart_Face_Analysis_System

An intelligent real-time face analysis system using Machine Learning and Computer Vision that detects face shape, skin tone, and recommends suitable hairstyles.

## 📋 Description
This system captures video from your webcam, detects faces using Haar Cascade classifier, and analyzes facial features using pre-trained ML models to provide personalized recommendations.

## ✨ Features
- Real-time face detection
- Face shape classification (Oval/Round/Square)
- Skin tone detection (Fair/Medium/Dark)
- Hairstyle recommendations based on face shape
- Multi-face detection with visual warning system
- Clean, user-friendly interface

## 🛠️ Technologies Used
- Python 3.x
- OpenCV (Computer Vision)
- NumPy (Numerical computations)
- joblib (Model loading)
- Scikit-learn (ML models)
- Haar Cascade (Face detection)

## 📁 Project Files
- `main.py` - Main application code
- `face_shape_model.pkl` - Face shape classification model
- `skin_model.pkl` - Skin tone detection model
- `hair_model.pkl` - Hairstyle recommendation model
- `haarcascade_frontalface_default.xml` - Face detection classifier

## 📦 Installation
1. Clone this repository
2. Install required packages:
   ```bash
   pip install opencv-python numpy joblib scikit-learn

# Run the application:

bash
python main.py

🎯 How It Works
Opens your webcam feed

Detects faces in real-time

Analyzes the main face for:

Face shape (using height/width ratio)

Skin tone (using average pixel intensity)

Recommended hairstyle (based on face shape)

Displays results on screen with visual feedback

⚠️ Multi-Face Detection
The system can detect multiple faces and displays:

Main face (green box) being analyzed

Additional faces (orange boxes)

Warning message when multiple faces are present

⌨️ Controls
Press ESC to exit the application

📝 License
This project is for educational purposes.

👨‍💻 Author
Kinjal Valji Boricha

🙏 Acknowledgments
OpenCV for computer vision tools

Haar Cascade for face detection

Scikit-learn for ML models

After pasting this, scroll down and click **"Commit changes"** .

Your repository will now have a professional-looking README that explains your project perfectly! Would you like help with anything else?
