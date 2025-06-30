from pushup_feedback import PushUpFeedback

model_path =r"C:\Users\Fatim\OneDrive\Desktop\BPR Submission\Final_pushup_cnn_Bilstm_model.keras"
feedback_system = PushUpFeedback(model_path)
feedback_system.analyze_from_webcam()