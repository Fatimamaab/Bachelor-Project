import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import mediapipe as mp

class PushUpFeedback:
    def __init__(self, model_path, max_frames=150):
        self.model = load_model(model_path)
        self.max_frames = max_frames
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=False)
        self.pose_connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12),
            (11, 13), (13, 15), (15, 17),
            (12, 14), (14, 16), (16, 18),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28),
            (27, 29), (29, 31), (28, 30), (30, 32)
        ]

    def analyze_from_npy(self, npy_path, sample_index=0, frame_index=50):
        data = np.load(npy_path, allow_pickle=True)
        pose_sequence = data[sample_index]  # shape: (150, 66)
        self._analyze(pose_sequence, frame_index)

    def analyze_from_video(self, video_path, frame_index=50):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        sequence = self._extract_pose_sequence(video_path)
        self._analyze(sequence, frame_index, video_path)

    def _extract_pose_sequence(self, video_path):
        cap = cv2.VideoCapture(video_path)
        keypoints_sequence = []

        print("Video opened:", cap.isOpened())
        print("Frame count:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        while cap.isOpened() and len(keypoints_sequence) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                keypoints = np.array([[lm.x, lm.y] for lm in landmarks]).flatten()
                if keypoints.shape[0] == 66:
                    keypoints_sequence.append(keypoints)

        cap.release()

        if len(keypoints_sequence) == 0:
            raise ValueError("No valid pose frames extracted.")
            
        return np.array(keypoints_sequence)

    def _analyze(self, pose_sequence, frame_index=0, video_path=None):
        print("Pose sequence shape:", pose_sequence.shape)

        # Flatten and reshape based on actual length
        sequence_flat = pose_sequence.reshape(1, -1)
        sequence_reshaped = sequence_flat.reshape(1, 1, sequence_flat.shape[1])
        prediction = self.model.predict(sequence_reshaped)

        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        print(f"\nPredicted Class: {['Incorrect', 'Correct'][predicted_class]} ({confidence:.2f} confidence)")

        angles, symmetry, velocity, feedback = self._generate_feedback(pose_sequence)

        print("\n--- Sample Feedback (first 10 frames) ---")
    
        for f in feedback[:10]:
            print(f)

        self._plot_metrics(angles, symmetry, velocity)

        # Handle video or blank frame
        if video_path:
            frame = self._extract_frame(video_path, frame_index)
        else:
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

        if frame_index >= pose_sequence.shape[0]:
            frame_index = pose_sequence.shape[0] - 1

        keypoints = pose_sequence[frame_index].reshape(33, 2)
        self._draw_pose_with_feedback(frame, keypoints, feedback[frame_index])

    def _generate_feedback(self, sequence):
        angles, symmetry, velocity, feedback = [], [], [], []
        LS, LE, LW = 11*2, 13*2, 15*2
        RS, RE, RW = 12*2, 14*2, 16*2

        for frame in sequence:
            l_ang = self._calculate_angle(frame[LS:LS+2], frame[LE:LE+2], frame[LW:LW+2])
            r_ang = self._calculate_angle(frame[RS:RS+2], frame[RE:RE+2], frame[RW:RW+2])
            angles.append([l_ang, r_ang])
            symmetry.append(abs(l_ang - r_ang))

        angles = np.array(angles)
        velocity = np.vstack((np.zeros((1, 2)), np.abs(np.diff(angles, axis=0))))

        for i in range(len(angles)):
            l, r = angles[i]
            sym = symmetry[i]
            vel = velocity[i]
            issues = []

            if l > 170:
                issues.append(f"Left elbow too extended ({l:.1f} degree), aim for ~110 degree")
            elif l < 80:
                issues.append(f"Left elbow too bent ({l:.1f} degree), aim for ~110 degree")

            if r > 170:
                issues.append(f"Right elbow too extended ({r:.1f} degree), aim for ~110 degree")
            elif r < 80:
                issues.append(f"Right elbow too bent ({r:.1f} degree), aim for ~110 degree")

            if sym > 15:
                issues.append("Uneven Form")

            if np.any(vel > 20):
                issues.append("Slow Down Your Movement")

            feedback.append(f"Frame {i+1}: {', '.join(issues) if issues else 'Good form'}")

        return angles, np.array(symmetry), velocity, feedback

    def _calculate_angle(self, p1, p2, p3):
        v1, v2 = np.array(p1) - np.array(p2), np.array(p3) - np.array(p2)
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.arccos(np.clip(cos_angle, -1.0, 1.0)) * (180 / np.pi)

    def _plot_metrics(self, angles, symmetry, velocity):
        plt.plot(angles[:, 0], label="Left Elbow")
        plt.plot(angles[:, 1], label="Right Elbow")
        plt.title("Elbow Angles Over Time")
        plt.legend(); plt.show()

        plt.plot(symmetry)
        plt.title("Posture Symmetry")
        plt.show()

        plt.plot(velocity[:, 0], label="Left Velocity")
        plt.plot(velocity[:, 1], label="Right Velocity")
        plt.title("Elbow Velocity Over Time")
        plt.legend(); plt.show()

    def _extract_frame(self, video_path, frame_num):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError("Could not extract frame.")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _draw_pose_with_feedback(self, frame, keypoints, feedback_text):
        frame_copy = frame.copy()
        h, w, _ = frame.shape

        # Draw keypoints (red)
        for x, y in keypoints:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame_copy, (cx, cy), 6, (0, 0, 255), -1)  # Red dots

        # Draw skeleton connections (green)
        for p1, p2 in self.pose_connections:
            if p1 < len(keypoints) and p2 < len(keypoints):
                x1, y1 = keypoints[p1]
                x2, y2 = keypoints[p2]
                pt1 = (int(x1 * w), int(y1 * h))
                pt2 = (int(x2 * w), int(y2 * h))
                cv2.line(frame_copy, pt1, pt2, (0, 255, 0), 2)

        # Feedback text in blue with background for readability
        y_offset = 30
        for i, line in enumerate(feedback_text.split(", ")):
            text_position = (10, y_offset + i * 25)
            cv2.putText(frame_copy, line, text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Show frame with Matplotlib
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
        plt.title("Pose with Feedback")
        plt.axis("off")
        plt.show()

    def analyze_from_webcam(self, max_frames=150):
        import matplotlib.pyplot as plt

        cap = cv2.VideoCapture(0)
        prev_angles = [0, 0]
        frame_count = 0

        print("Starting real-time feedback... Close the plot window to exit.")

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            frame_display = frame.copy()
            h, w, _ = frame.shape

            feedback_text = "No pose detected"
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                keypoints = np.array([[lm.x, lm.y] for lm in landmarks])
                if keypoints.shape[0] == 33:
                    LS, LE, LW = keypoints[11], keypoints[13], keypoints[15]
                    RS, RE, RW = keypoints[12], keypoints[14], keypoints[16]

                    l_ang = self._calculate_angle(LS, LE, LW)
                    r_ang = self._calculate_angle(RS, RE, RW)
                    symmetry = abs(l_ang - r_ang)
                    velocity = [abs(l_ang - prev_angles[0]), abs(r_ang - prev_angles[1])]
                    prev_angles = [l_ang, r_ang]

                    issues = []
                    if l_ang > 170:
                        issues.append(f"Left elbow overextended ({l_ang:.1f} degree)")
                    elif l_ang < 80:
                        issues.append(f"Left elbow too bent ({l_ang:.1f} degree)")

                    if r_ang > 170:
                        issues.append(f"Right elbow overextended ({r_ang:.1f} degree)")
                    elif r_ang < 80:
                        issues.append(f"Right elbow too bent ({r_ang:.1f} degree)")

                    if symmetry > 15:
                        issues.append("Uneven Form")
                    if any(v > 20 for v in velocity):
                        issues.append("Slow Down Your Movement")

                    feedback_text = ", ".join(issues) if issues else "Good form"

                    for x, y in keypoints:
                        cv2.circle(frame_display, (int(x * w), int(y * h)), 5, (0, 0, 255), -1)
                    for p1, p2 in self.pose_connections:
                        x1, y1 = keypoints[p1]
                        x2, y2 = keypoints[p2]
                        pt1 = (int(x1 * w), int(y1 * h))
                        pt2 = (int(x2 * w), int(y2 * h))
                        cv2.line(frame_display, pt1, pt2, (0, 255, 0), 2)


            # Overlay feedback text
            y_offset = 30
            for i, line in enumerate(feedback_text.split(", ")):
                pos = (10, y_offset + i * 25)
                cv2.putText(frame_display, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)


            # Display using matplotlib
            plt.imshow(cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB))
            plt.title(f"Frame {frame_count+1}")
            plt.axis("off")
            plt.pause(0.001)
            plt.clf()


            frame_count += 1

        cap.release()
        plt.close()
        print("Feedback session ended.")
