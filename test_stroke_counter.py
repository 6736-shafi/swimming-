#!/usr/bin/env python3
"""
Test script to verify the stroke counting functionality
without the Streamlit UI
"""
import cv2
import mediapipe as mp
import sys

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class StrokeCounter:
    def __init__(self):
        self.left_strokes = 0
        self.right_strokes = 0
        
        # Track previous positions for velocity calculation
        self.left_prev_y = None
        self.right_prev_y = None
        self.left_prev_velocity = 0
        self.right_prev_velocity = 0
        
        # State tracking: looking for recovery (up) then entry+pull (down)
        self.left_state = "neutral"  # "recovery", "entry", "neutral"
        self.right_state = "neutral"
        
        # Cooldown to prevent double counting (frames)
        self.left_cooldown = 0
        self.right_cooldown = 0
        
    def detect_stroke(self, landmarks):
        """
        Improved stroke detection using position tracking and velocity.
        
        Freestyle stroke cycle:
        1. Recovery: Hand moves forward above water (wrist above shoulder, moving up)
        2. Entry: Hand enters water and extends forward (wrist moving down)
        3. Pull: Hand pulls through water (wrist below shoulder)
        
        We count a stroke when the full cycle completes: recovery -> entry -> pull
        """
        if not landmarks:
            return self.left_strokes + self.right_strokes
        
        # Get key landmarks
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        
        # Calculate vertical difference (relative to shoulder)
        left_y_diff = left_wrist.y - left_shoulder.y
        right_y_diff = right_wrist.y - right_shoulder.y
        
        # Calculate velocity (change in y position)
        left_velocity = 0
        right_velocity = 0
        
        if self.left_prev_y is not None:
            left_velocity = left_y_diff - self.left_prev_y
        if self.right_prev_y is not None:
            right_velocity = right_y_diff - self.right_prev_y
        
        # Update cooldowns
        if self.left_cooldown > 0:
            self.left_cooldown -= 1
        if self.right_cooldown > 0:
            self.right_cooldown -= 1
        
        # LEFT ARM DETECTION
        if self.left_cooldown == 0:
            # Recovery phase: wrist is above shoulder
            if left_y_diff < -0.11 and left_velocity < -0.01:  # Precise tuning
                self.left_state = "recovery"
            
            # Entry/Pull phase: wrist moving down after recovery
            elif self.left_state == "recovery" and left_y_diff > 0.04:  # Precise threshold
                # Complete stroke detected!
                self.left_strokes += 1
                self.left_state = "neutral"
                self.left_cooldown = 13  # Precise cooldown
                print(f"  [LEFT STROKE] Total left: {self.left_strokes}")
        
        # RIGHT ARM DETECTION  
        if self.right_cooldown == 0:
            # Recovery phase: wrist is above shoulder
            if right_y_diff < -0.11 and right_velocity < -0.01:  # Precise tuning
                self.right_state = "recovery"
            
            # Entry/Pull phase: wrist moving down after recovery
            elif self.right_state == "recovery" and right_y_diff > 0.04:  # Precise threshold
                # Complete stroke detected!
                self.right_strokes += 1
                self.right_state = "neutral"
                self.right_cooldown = 13  # Precise cooldown
                print(f"  [RIGHT STROKE] Total right: {self.right_strokes}")        
        # Store current positions for next frame
        self.left_prev_y = left_y_diff
        self.right_prev_y = right_y_diff
        self.left_prev_velocity = left_velocity
        self.right_prev_velocity = right_velocity
        
        return self.left_strokes + self.right_strokes

def test_stroke_counter(video_path, output_path=None):
    """Process video and count swimming strokes."""
    
    print(f"Processing video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video properties:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - FPS: {fps}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Duration: {duration:.2f}s")
    print()
    
    # Setup output if requested
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output will be saved to: {output_path}\n")
    
    # Initialize pose detector and stroke counter
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    counter = StrokeCounter()
    frame_count = 0
    
    print("Processing frames...\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = pose.process(rgb_frame)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Count strokes
            stroke_count = counter.detect_stroke(results.pose_landmarks.landmark)
        else:
            stroke_count = counter.left_strokes + counter.right_strokes
        
        # Add overlay
        cv2.putText(frame, f"Strokes: {stroke_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Left: {counter.left_strokes} | Right: {counter.right_strokes}",
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Write frame if output requested
        if out:
            out.write(frame)
        
        # Progress indicator
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames) - Strokes: {stroke_count}")
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    pose.close()
    
    total_strokes = counter.left_strokes + counter.right_strokes
    stroke_rate = (total_strokes / duration * 60) if duration > 0 else 0
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Total Strokes: {total_strokes}")
    print(f"  - Left arm: {counter.left_strokes}")
    print(f"  - Right arm: {counter.right_strokes}")
    print(f"Video Duration: {duration:.2f}s")
    print(f"Stroke Rate: {stroke_rate:.1f} strokes/minute")
    print("="*60)
    
    return total_strokes

if __name__ == "__main__":
    video_path = "/Users/mdshafiuddin/Desktop/us-swimming/02 (3).mp4"
    output_path = "/Users/mdshafiuddin/Desktop/us-swimming/processed_02_3.mp4"
    
    test_stroke_counter(video_path, output_path)
