"""
Swimming Analytics Module
Provides predictive analytics for swimming performance including:
- Stroke timing prediction
- Finish time estimation
- Fatigue detection
- Pace analysis
- Performance scoring
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

class SwimmerAnalytics:
    """Analyzes swimming performance and provides predictive insights."""
    
    def __init__(self, pool_length: float = 50.0, race_distance: Optional[float] = None):
        """
        Initialize swimmer analytics.
        
        Args:
            pool_length: Length of pool in meters (default: 50m)
            race_distance: Total race distance in meters (optional)
        """
        self.pool_length = pool_length
        self.race_distance = race_distance or pool_length
        
        # Stroke tracking
        self.stroke_times = []  # Timestamp of each stroke
        self.stroke_intervals = []  # Time between consecutive strokes
        self.left_stroke_times = []
        self.right_stroke_times = []
        
        # Performance metrics
        self.start_time = 0
        self.current_time = 0
        
    def add_stroke(self, timestamp: float, arm: str = 'both'):
        """
        Record a stroke occurrence.
        
        Args:
            timestamp: Time in seconds when stroke occurred
            arm: Which arm ('left', 'right', or 'both')
        """
        self.stroke_times.append(timestamp)
        
        if arm == 'left':
            self.left_stroke_times.append(timestamp)
        elif arm == 'right':
            self.right_stroke_times.append(timestamp)
            
        # Calculate interval if we have previous stroke
        if len(self.stroke_times) > 1:
            interval = timestamp - self.stroke_times[-2]
            self.stroke_intervals.append(interval)
            
        self.current_time = timestamp
        
        if self.start_time == 0:
            self.start_time = timestamp
    
    def predict_next_stroke(self, window_size: int = 5) -> Tuple[float, float]:
        """
        Predict when the next stroke will occur.
        
        Args:
            window_size: Number of recent strokes to consider
            
        Returns:
            (predicted_time, confidence): Time of next stroke and confidence (0-1)
        """
        if len(self.stroke_intervals) < 2:
            return (self.current_time + 2.0, 0.0)
        
        # Use moving average of recent intervals
        recent_intervals = self.stroke_intervals[-window_size:]
        avg_interval = np.mean(recent_intervals)
        std_interval = np.std(recent_intervals)
        
        # Confidence based on consistency (lower std = higher confidence)
        confidence = max(0.0, 1.0 - (std_interval / avg_interval if avg_interval > 0 else 1.0))
        
        predicted_time = self.current_time + avg_interval
        
        return (predicted_time, confidence)
    
    def estimate_stroke_length(self) -> float:
        """
        Estimate average distance covered per stroke.
        
        Returns:
            Average stroke length in meters
        """
        if not self.stroke_times:
            return 2.0  # Default assumption
        
        # Rough estimate: pool_length / strokes_per_length
        # Assume average of 20-25 strokes per 50m for recreational swimmers
        return self.pool_length / 22.0
    
    def estimate_current_position(self) -> float:
        """
        Estimate current distance swum.
        
        Returns:
            Distance in meters
        """
        total_strokes = len(self.stroke_times)
        avg_stroke_length = self.estimate_stroke_length()
        return total_strokes * avg_stroke_length
    
    def predict_finish_time(self) -> Tuple[float, str]:
        """
        Predict race finish time based on current pace.
        
        Returns:
            (finish_time, confidence_level): Time in seconds and confidence description
        """
        if not self.stroke_times or len(self.stroke_times) < 5:
            return (0.0, "Insufficient data")
        
        # Calculate current pace
        elapsed_time = self.current_time - self.start_time
        current_position = self.estimate_current_position()
        
        if current_position <= 0:
            return (0.0, "Insufficient data")
        
        current_pace = current_position / elapsed_time  # meters per second
        
        # Estimate remaining distance and time
        remaining_distance = max(0, self.race_distance - current_position)
        remaining_time = remaining_distance / current_pace if current_pace > 0 else 0
        
        finish_time = elapsed_time + remaining_time
        
        # Confidence based on data completeness
        completion_pct = (current_position / self.race_distance) * 100
        if completion_pct < 20:
            confidence = "Low"
        elif completion_pct < 50:
            confidence = "Medium"
        else:
            confidence = "High"
        
        return (finish_time, confidence)
    
    def calculate_fatigue_index(self) -> Tuple[float, str]:
        """
        Calculate fatigue index by comparing first half vs second half pace.
        
        Returns:
            (fatigue_index, status): Percentage slowdown and status description
        """
        if len(self.stroke_intervals) < 6:
            return (0.0, "Insufficient data")
        
        mid_point = len(self.stroke_intervals) // 2
        
        first_half_avg = np.mean(self.stroke_intervals[:mid_point])
        second_half_avg = np.mean(self.stroke_intervals[mid_point:])
        
        # Positive index = slowing down, negative = speeding up
        fatigue_index = ((second_half_avg - first_half_avg) / first_half_avg) * 100
        
        # Categorize fatigue
        if fatigue_index < -5:
            status = "Negative Split (Excellent!)"
        elif fatigue_index < 5:
            status = "Consistent (Good)"
        elif fatigue_index < 15:
            status = "Moderate Fatigue"
        else:
            status = "High Fatigue"
        
        return (fatigue_index, status)
    
    def get_pace_breakdown(self, segment_size: int = 5) -> List[Dict]:
        """
        Break down pace by segments of strokes.
        
        Args:
            segment_size: Number of strokes per segment
            
        Returns:
            List of segments with pace data
        """
        if len(self.stroke_intervals) < segment_size:
            return []
        
        segments = []
        num_segments = len(self.stroke_intervals) // segment_size
        
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size
            
            segment_intervals = self.stroke_intervals[start_idx:end_idx]
            avg_interval = np.mean(segment_intervals)
            
            # Calculate stroke rate (strokes per minute)
            stroke_rate = 60.0 / avg_interval if avg_interval > 0 else 0
            
            segments.append({
                'segment': i + 1,
                'strokes': f"{start_idx + 1}-{end_idx}",
                'avg_interval': avg_interval,
                'stroke_rate': stroke_rate
            })
        
        return segments
    
    def calculate_performance_score(self) -> Tuple[int, str]:
        """
        Calculate overall performance score (0-100).
        
        Factors:
        - Stroke efficiency (consistency of intervals)
        - Speed (compared to benchmarks)
        - Fatigue resistance
        
        Returns:
            (score, rating): Score 0-100 and rating description
        """
        if len(self.stroke_intervals) < 5:
            return (0, "Insufficient data")
        
        # 1. Consistency score (40 points) - lower variance is better
        interval_std = np.std(self.stroke_intervals)
        interval_mean = np.mean(self.stroke_intervals)
        cv = interval_std / interval_mean if interval_mean > 0 else 1.0  # Coefficient of variation
        consistency_score = max(0, 40 - (cv * 100))
        
        # 2. Speed score (40 points) - based on stroke rate
        stroke_rate = 60.0 / interval_mean if interval_mean > 0 else 0
        # Elite: 45+, Competitive: 35-45, Recreational: 25-35, Beginner: <25
        if stroke_rate >= 45:
            speed_score = 40
        elif stroke_rate >= 35:
            speed_score = 30 + ((stroke_rate - 35) / 10) * 10
        elif stroke_rate >= 25:
            speed_score = 20 + ((stroke_rate - 25) / 10) * 10
        else:
            speed_score = (stroke_rate / 25) * 20
        
        # 3. Fatigue resistance (20 points)
        fatigue_index, _ = self.calculate_fatigue_index()
        if fatigue_index < 0:  # Negative split
            fatigue_score = 20
        elif fatigue_index < 5:
            fatigue_score = 15
        elif fatigue_index < 15:
            fatigue_score = 10
        else:
            fatigue_score = max(0, 10 - fatigue_index)
        
        total_score = int(consistency_score + speed_score + fatigue_score)
        
        # Rating
        if total_score >= 85:
            rating = "Elite"
        elif total_score >= 70:
            rating = "Competitive"
        elif total_score >= 50:
            rating = "Recreational"
        else:
            rating = "Beginner"
        
        return (total_score, rating)
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics of the swimming performance."""
        if not self.stroke_times:
            return {}
        
        total_strokes = len(self.stroke_times)
        elapsed_time = self.current_time - self.start_time if self.start_time > 0 else 0
        
        avg_interval = np.mean(self.stroke_intervals) if self.stroke_intervals else 0
        stroke_rate = 60.0 / avg_interval if avg_interval > 0 else 0
        
        estimated_distance = self.estimate_current_position()
        avg_pace = elapsed_time / estimated_distance if estimated_distance > 0 else 0  # sec/meter
        
        return {
            'total_strokes': total_strokes,
            'left_strokes': len(self.left_stroke_times),
            'right_strokes': len(self.right_stroke_times),
            'elapsed_time': elapsed_time,
            'stroke_rate': stroke_rate,
            'avg_interval': avg_interval,
            'estimated_distance': estimated_distance,
            'avg_pace': avg_pace
        }
