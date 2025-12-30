import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from analytics import SwimmerAnalytics

# Configure page
st.set_page_config(
    page_title="Swimming Analytics",
    page_icon="🏊",
    layout="wide"
)

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
        
        # State tracking
        self.left_state = "neutral"
        self.right_state = "neutral"
        
        # Cooldown to prevent double counting
        self.left_cooldown = 0
        self.right_cooldown = 0
        
        # Stroke timing for analytics
        self.left_stroke_times = []
        self.right_stroke_times = []
        
    def detect_stroke(self, landmarks, timestamp):
        """Detect strokes and track timing."""
        if not landmarks:
            return self.left_strokes + self.right_strokes
        
        # Get key landmarks
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # Calculate vertical difference
        left_y_diff = left_wrist.y - left_shoulder.y
        right_y_diff = right_wrist.y - right_shoulder.y
        
        # Calculate velocity
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
            if left_y_diff < -0.11 and left_velocity < -0.01:
                self.left_state = "recovery"
            elif self.left_state == "recovery" and left_y_diff > 0.04:
                self.left_strokes += 1
                self.left_stroke_times.append(timestamp)
                self.left_state = "neutral"
                self.left_cooldown = 13
        
        # RIGHT ARM DETECTION  
        if self.right_cooldown == 0:
            if right_y_diff < -0.11 and right_velocity < -0.01:
                self.right_state = "recovery"
            elif self.right_state == "recovery" and right_y_diff > 0.04:
                self.right_strokes += 1
                self.right_stroke_times.append(timestamp)
                self.right_state = "neutral"
                self.right_cooldown = 13
        
        # Store current positions
        self.left_prev_y = left_y_diff
        self.right_prev_y = right_y_diff
        self.left_prev_velocity = left_velocity
        self.right_prev_velocity = right_velocity
        
        return self.left_strokes + self.right_strokes

def process_video_with_analytics(video_path, progress_bar, status_text, pool_length=50.0, race_distance=None):
    """Process video and count swimming strokes with analytics."""
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file")
        return None, None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize analytics
    analytics = SwimmerAnalytics(pool_length=pool_length, race_distance=race_distance)
    
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
    
    status_text.text("Processing video...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps  # Time in seconds
        
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
            prev_total = counter.left_strokes + counter.right_strokes
            stroke_count = counter.detect_stroke(results.pose_landmarks.landmark, timestamp)
            
            # If new stroke detected, add to analytics
            if stroke_count > prev_total:
                # Determine which arm
                if len(counter.left_stroke_times) > 0 and counter.left_stroke_times[-1] == timestamp:
                    analytics.add_stroke(timestamp, 'left')
                elif len(counter.right_stroke_times) > 0 and counter.right_stroke_times[-1] == timestamp:
                    analytics.add_stroke(timestamp, 'right')
        else:
            stroke_count = counter.left_strokes + counter.right_strokes
        
        # Add stroke count overlay
        cv2.putText(frame, f"Strokes: {stroke_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Left: {counter.left_strokes} | Right: {counter.right_strokes}",
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Write frame
        out.write(frame)
        
        # Update progress
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames} - Strokes: {stroke_count}")
    
    # Cleanup
    cap.release()
    out.release()
    pose.close()
    
    return output_path, analytics

def main():
    # Header
    st.title("🏊 Swimming Analytics & Stroke Counter")
    st.markdown("""
    Upload a swimming video for comprehensive analysis including stroke counting, 
    performance predictions, and fatigue detection!
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Pool Settings")
        pool_length = st.selectbox(
            "Pool Length",
            options=[25, 50, 25],
            format_func=lambda x: f"{x}m (Olympic)" if x == 50 else (f"{x}m (Short Course)" if x == 25 else f"{x}yd"),
            index=1,
            help="Select your pool size"
        )
        
        race_distance_option = st.selectbox(
            "Race Distance",
            options=["Auto-detect", "50m", "100m", "200m", "Custom"],
            help="Total distance being swum"
        )
        
        if race_distance_option == "Custom":
            race_distance = st.number_input("Enter distance (meters)", min_value=25, max_value=1500, value=100)
        elif race_distance_option != "Auto-detect":
            race_distance = int(race_distance_option.replace("m", ""))
        else:
            race_distance = pool_length
        
        st.divider()
        
        st.header("ℹ️ About")
        st.info("""
        **Features:**
        - Stroke counting with AI
        - Next stroke prediction
        - Finish time estimation
        - Fatigue analysis
        - Performance scoring
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📹 Upload Swimming Video",
        type=['mp4', 'avi', 'mov'],
        help="Upload a side-view video of freestyle swimming"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        # Display original video
        st.subheader("📹 Original Video")
        st.video(video_path)
        
        # Process button
        if st.button("🔄 Process & Analyze", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process video
            with st.spinner("Processing video with AI analytics..."):
                output_path, analytics = process_video_with_analytics(
                    video_path, progress_bar, status_text, pool_length, race_distance
                )
            
            if output_path and analytics:
                st.success("✅ Analysis Complete!")
                
                # Get analytics data
                stats = analytics.get_summary_stats()
                next_stroke_time, next_stroke_conf = analytics.predict_next_stroke()
                finish_time, finish_conf = analytics.predict_finish_time()
                fatigue_idx, fatigue_status = analytics.calculate_fatigue_index()
                perf_score, perf_rating = analytics.calculate_performance_score()
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Predictions", "📉 Fatigue Analysis", "⭐ Performance"])
                
                with tab1:
                    # Display processed video
                    st.subheader("🎯 Processed Video")
                    st.video(output_path)
                    
                    # Download button
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            "⬇️ Download Processed Video",
                            f, file_name=f"analyzed_{uploaded_file.name}",
                            mime="video/mp4", use_container_width=True
                        )
                    
                    # Key metrics
                    st.subheader("📈 Key Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Strokes", stats.get('total_strokes', 0))
                    with col2:
                        st.metric("Stroke Rate", f"{stats.get('stroke_rate', 0):.1f}/min")
                    with col3:
                        st.metric("Duration", f"{stats.get('elapsed_time', 0):.1f}s")
                    with col4:
                        st.metric("Est. Distance", f"{stats.get('estimated_distance', 0):.1f}m")
                
                with tab2:
                    st.subheader("🔮 Performance Predictions")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Next Stroke In",
                            f"{max(0, next_stroke_time - stats.get('elapsed_time', 0)):.1f}s",
                            help=f"Confidence: {next_stroke_conf*100:.0f}%"
                        )
                    with col2:
                        if finish_time > 0:
                            st.metric(
                                f"Predicted Finish ({race_distance}m)",
                                f"{finish_time:.1f}s",
                                help=f"Confidence: {finish_conf}"
                            )
                    
                    # Pace breakdown
                    pace_data = analytics.get_pace_breakdown()
                    if pace_data:
                        st.subheader("📊 Pace by Segment")
                        df = pd.DataFrame(pace_data)
                        fig = px.bar(df, x='segment', y='stroke_rate',
                                    labels={'segment': 'Segment', 'stroke_rate': 'Stroke Rate (strokes/min)'},
                                    title="Stroke Rate by Segment")
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    st.subheader("📉 Fatigue Detection")
                    
                    fatigue_color = "🟢" if fatigue_idx < 5 else ("🟡" if fatigue_idx < 15 else "🔴")
                    st.metric(
                        "Fatigue Index",
                        f"{fatigue_color} {fatigue_idx:.1f}%",
                        delta=fatigue_status
                    )
                    
                    # Stroke intervals chart
                    if analytics.stroke_intervals:
                        st.subheader("⏱️ Stroke Interval Progression")
                        intervals_df = pd.DataFrame({
                            'Stroke': range(1, len(analytics.stroke_intervals) + 1),
                            'Interval (seconds)': analytics.stroke_intervals
                        })
                        fig = px.line(intervals_df, x='Stroke', y='Interval (seconds)',
                                     title="Time Between Strokes (Shows Fatigue)",
                                     markers=True)
                        fig.add_hline(y=np.mean(analytics.stroke_intervals), 
                                     line_dash="dash", line_color="red",
                                     annotation_text="Average")
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab4:
                    st.subheader("⭐ Performance Score")
                    
                    # Gauge chart for score
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = perf_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"Performance Rating: {perf_rating}"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps' : [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 70], 'color': "lightyellow"},
                                {'range': [70, 85], 'color': "lightgreen"},
                                {'range': [85, 100], 'color': "lightblue"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed breakdown
                    st.subheader("📋 Detailed Statistics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Stroke Count:**")
                        st.write(f"- Left: {stats.get('left_strokes', 0)}")
                        st.write(f"- Right: {stats.get('right_strokes', 0)}")
                        st.write(f"- Balance: {abs(stats.get('left_strokes', 0) - stats.get('right_strokes', 0))}")
                    with col2:
                        st.write("**Timing:**")
                        st.write(f"- Avg Interval: {stats.get('avg_interval', 0):.2f}s")
                        st.write(f"- Stroke Rate: {stats.get('stroke_rate', 0):.1f}/min")
                        st.write(f"- Avg Pace: {stats.get('avg_pace', 0):.2f}s/m")
                
                # Cleanup
                try:
                    os.unlink(output_path)
                except:
                    pass
        
        # Cleanup uploaded file
        try:
            os.unlink(video_path)
        except:
            pass
    else:
        st.info("👆 Upload a swimming video to get started with comprehensive analytics!")
        
        st.subheader("✨ What You'll Get")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Stroke Analysis:**")
            st.write("- Accurate stroke counting")
            st.write("- Left/right arm balance")
            st.write("- Stroke rate tracking")
        with col2:
            st.write("**Predictive Analytics:**")
            st.write("- Next stroke prediction")
            st.write("- Finish time estimation")
            st.write("- Performance scoring")
            st.write("- Fatigue detection")

if __name__ == "__main__":
    main()
