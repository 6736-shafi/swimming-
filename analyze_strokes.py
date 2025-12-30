#!/usr/bin/env python3
"""
Debug script to analyze swimming strokes and determine the actual count
This will help us understand what's happening in each frame
"""
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def analyze_strokes(video_path):
    """Analyze video frame by frame to understand stroke patterns."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video: {duration:.1f}s, {total_frames} frames, {fps} fps\n")
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    frame_count = 0
    left_wrist_positions = []
    right_wrist_positions = []
    
    print("Sampling wrist positions every 10 frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Sample every 10 frames
        if frame_count % 10 != 0:
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            timestamp = frame_count / fps
            
            # Calculate relative positions
            left_y_diff = left_wrist.y - left_shoulder.y
            right_y_diff = right_wrist.y - right_shoulder.y
            left_x_diff = left_wrist.x - left_shoulder.x
            right_x_diff = right_wrist.x - right_shoulder.x
            
            left_wrist_positions.append({
                'frame': frame_count,
                'time': timestamp,
                'y_diff': left_y_diff,
                'x_diff': left_x_diff,
                'visibility': left_wrist.visibility
            })
            
            right_wrist_positions.append({
                'frame': frame_count,
                'time': timestamp,
                'y_diff': right_y_diff,
                'x_diff': right_x_diff,
                'visibility': right_wrist.visibility
            })
    
    cap.release()
    pose.close()
    
    # Analyze patterns
    print("\n" + "="*80)
    print("LEFT ARM ANALYSIS")
    print("="*80)
    print(f"{'Frame':<8} {'Time(s)':<8} {'Y-Diff':<10} {'X-Diff':<10} {'Phase':<15}")
    print("-"*80)
    
    left_strokes = 0
    prev_y = None
    
    for i, pos in enumerate(left_wrist_positions):
        # Determine phase
        if pos['y_diff'] < -0.1:
            phase = "Recovery (up)"
        elif pos['y_diff'] > 0.1:
            phase = "Pull (down)"
        else:
            phase = "Transition"
        
        # Count strokes - detect transitions from up to down
        if prev_y is not None:
            if prev_y < -0.05 and pos['y_diff'] > 0.05:
                left_strokes += 1
                phase += " [STROKE!]"
        
        print(f"{pos['frame']:<8} {pos['time']:<8.1f} {pos['y_diff']:<10.3f} {pos['x_diff']:<10.3f} {phase:<15}")
        prev_y = pos['y_diff']
    
    print(f"\nDetected LEFT strokes: {left_strokes}")
    
    print("\n" + "="*80)
    print("RIGHT ARM ANALYSIS")
    print("="*80)
    print(f"{'Frame':<8} {'Time(s)':<8} {'Y-Diff':<10} {'X-Diff':<10} {'Phase':<15}")
    print("-"*80)
    
    right_strokes = 0
    prev_y = None
    
    for i, pos in enumerate(right_wrist_positions):
        if pos['y_diff'] < -0.1:
            phase = "Recovery (up)"
        elif pos['y_diff'] > 0.1:
            phase = "Pull (down)"
        else:
            phase = "Transition"
        
        if prev_y is not None:
            if prev_y < -0.05 and pos['y_diff'] > 0.05:
                right_strokes += 1
                phase += " [STROKE!]"
        
        print(f"{pos['frame']:<8} {pos['time']:<8.1f} {pos['y_diff']:<10.3f} {pos['x_diff']:<10.3f} {phase:<15}")
        prev_y = pos['y_diff']
    
    print(f"\nDetected RIGHT strokes: {right_strokes}")
    print(f"\nTOTAL ESTIMATED STROKES: {left_strokes + right_strokes}")

if __name__ == "__main__":
    analyze_strokes("/Users/mdshafiuddin/Desktop/us-swimming/02 (3).mp4")
