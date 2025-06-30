# Push-Up Form Classification and Feedback System
### Overview
This project presents a real-time, interpretable feedback system for human physical activity, specifically focused on evaluating push-up form. By integrating 2D pose estimation, deep learning classification, and biomechanical feature analysis, the system classifies push-up sequences as correct or incorrect and provides frame-level feedback on common form deviations.

Developed as part of a Bachelor's final project, the system aims to assist users in improving their push-up technique autonomously through visual and textual guidance.

### Features

* Pose Estimation using MediaPipe Pose (33 keypoints per frame)
* Feature Engineering: Elbow Joint Angles, Posture Symmetry, and Movement Velocity.
* Hybrid CNN-BiLSTM Model for push-up classification
* Real-Time Frame-Level Feedback:
        Detection of overextended or insufficiently bent elbows, 
        Symmetry imbalance detection, and
        Sudden/uncontrolled movement detection.
* Deployment-Ready Export: Keras Model (.keras format) and TensorFlow Lite Model (.tflite format)
* Real-World Validation on external YouTube exercise videos
* Interactive Visualizations (pose skeletons and overlaid feedback)

### Project Structure

* pushup-feedback-system:
*     ── pushup_feedback.py                      # Core feedback engine (pose analysis + classification + feedback)
      ── run_feedback.py                         # Real-time feedback interface using webcam
* models:
*     ── pushup_cnn_lstm_model.keras             # Trained Keras model
      ── pushup_cnn_lstm_model.tflite            # TFLite model for mobile deployment
* images                                   # Sample visualizations (pose skeletons, elbow angles, velocity)
* notebooks:
*     ── 3 Python Notebooks                # Programming Solution
* README.md                                # Project documentation
* requirements.txt                         # Python dependencies

### Installation
1. Clone the repository:
      git clone https://github.com/your-username/pushup-feedback-system.git
      cd pushup-feedback-system

3. Install the dependencies:
      pip install -r requirements.txt

4. Run real-time feedback:
      python run_feedback.py

### Requirements
      Python 3.8+
      TensorFlow 2.x
      OpenCV
      MediaPipe
      NumPy
      Matplotlib
      scikit-learn
(Exact versions are listed in requirements.txt)

### Results
* Test Accuracy: 95.00%
* Test Loss: 0.0598
* F1-Score: 0.95 (both correct and incorrect classes)
* Generalization: Successfully validated on real-world YouTube exercise videos.

### Research Context
This work was developed as a final-year Bachelor's project at Noroff University College under the supervision of Shahnila Raheem.
It contributes to the growing field of AI-driven fitness assessment by combining deep learning, pose estimation, and explainable feedback mechanisms.

### License
This project is intended for academic and research purposes. Please cite appropriately if using or extending the system.

### Acknowledgements
* Supervisor: Shahnila Raheem (for invaluable guidance and support)
* MediaPipe by Google: Pose estimation framework
* TensorFlow and Keras: Deep learning frameworks
* Real-world push-up dataset references (external public videos for validation)

### Keywords
Pose Estimation, Push-Up Classification, CNN-BiLSTM, Biomechanical Feedback, Real-Time Feedback
