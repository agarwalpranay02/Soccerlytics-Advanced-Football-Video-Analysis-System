import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv
import pickle
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ----------------------------- UTILITY FUNCTIONS --------------------------------

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox):
    return bbox[2] - bbox[0]

def measure_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def measure_xy_distance(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

# ----------------------------- TRACKING MODULE --------------------------------

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for obj_type, obj_tracks in tracks.items():
            for frame_num, track in enumerate(obj_tracks):
                for track_id, info in track.items():
                    bbox = info['bbox']
                    pos = get_center_of_bbox(bbox) if obj_type == 'ball' else get_foot_position(bbox)
                    info['position'] = pos

    def interpolate_ball_positions(self, ball_frames):
        data = [x.get(1, {}).get('bbox', []) for x in ball_frames]
        df = pd.DataFrame(data, columns=['x1', 'y1', 'x2', 'y2'])
        df.interpolate(inplace=True)
        df.bfill(inplace=True)
        return [{1: {"bbox": row}} for row in df.to_numpy().tolist()]

    def detect_frames(self, frames):
        result = []
        for i in range(0, len(frames), 20):
            result.extend(self.model.predict(frames[i:i+20], conf=0.1))
        return result

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        tracks = {"players": [], "referees": [], "ball": []}

        for i, det in enumerate(detections):
            cls_map = det.names
            rev_map = {v: k for k, v in cls_map.items()}
            det_sv = sv.Detections.from_ultralytics(det)
            for j, cid in enumerate(det_sv.class_id):
                if cls_map[cid] == "goalkeeper":
                    det_sv.class_id[j] = rev_map['player']
            result = self.tracker.update_with_detections(det_sv)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for d in result:
                bbox, cid, tid = d[0].tolist(), d[3], d[4]
                if cid == rev_map['player']:
                    tracks['players'][i][tid] = {'bbox': bbox}
                elif cid == rev_map['referee']:
                    tracks['referees'][i][tid] = {'bbox': bbox}

            for d in det_sv:
                bbox, cid = d[0].tolist(), d[3]
                if cid == rev_map['ball']:
                    tracks['ball'][i][1] = {'bbox': bbox}

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_annotations(self, frames, tracks, team_control, view_transformer):
        output = []
        for i, frame in enumerate(frames):
            draw = frame.copy()
            
            # Draw annotations
            for pid, player in tracks['players'][i].items():
                color = player.get('team_color', (0, 0, 255))
                draw = self._draw_ellipse(draw, player['bbox'], color, pid)
                if player.get('has_ball', False):
                    draw = self._draw_triangle(draw, player['bbox'], (0, 0, 255))
            for ref in tracks['referees'][i].values():
                draw = self._draw_ellipse(draw, ref['bbox'], (0, 255, 255))
            for ball in tracks['ball'][i].values():
                draw = self._draw_triangle(draw, ball['bbox'], (0, 255, 0))
            
            # Draw possession bar
            draw = self._draw_possession_bar(draw, i, team_control)
            
            # Add 2D view
            if view_transformer is not None:
                draw = view_transformer.draw_2d_view_on_frame(draw, tracks, i)
            
            output.append(draw)
        return output

    def _draw_ellipse(self, frame, bbox, color, pid=None):
        x, y2 = get_center_of_bbox(bbox)[0], int(bbox[3])
        w = get_bbox_width(bbox)
        cv2.ellipse(frame, (x, y2), (int(w), int(0.35 * w)), 0, -45, 235, color, 2)
        if pid is not None:
            cv2.rectangle(frame, (x - 20, y2 + 5), (x + 20, y2 + 25), color, -1)
            cv2.putText(frame, str(pid), (x - 10, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame

    def _draw_triangle(self, frame, bbox, color):
        x, y = get_center_of_bbox(bbox)
        pts = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [pts], 0, color, -1)
        cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2)
        return frame

    def _draw_possession_bar(self, frame, idx, control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Calculate possession stats
        control_array = np.array(control)
        c1 = (control_array[:idx+1] == 1).sum()
        c2 = (control_array[:idx+1] == 2).sum()
        total = c1 + c2
        p1 = (c1 / total) * 100 if total else 0
        p2 = (c2 / total) * 100 if total else 0
        cv2.putText(frame, f"Team 1 Ball Control: {p1:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {p2:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        return frame

# ----------------------------- TEAM ASSIGNMENT MODULE --------------------------------

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        model = KMeans(n_clusters=2, init="k-means++", n_init=1)
        model.fit(image_2d)
        return model

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half = image[0:int(image.shape[0]/2), :]
        model = self.get_clustering_model(top_half)
        labels = model.labels_.reshape(top_half.shape[0], top_half.shape[1])
        corners = [labels[0,0], labels[0,-1], labels[-1,0], labels[-1,-1]]
        non_player_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 - non_player_cluster
        return model.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, players):
        colors = []
        for _, p in players.items():
            colors.append(self.get_player_color(frame, p['bbox']))
        model = KMeans(n_clusters=2, init="k-means++", n_init=1)
        model.fit(colors)
        self.kmeans = model
        self.team_colors[1] = model.cluster_centers_[0]
        self.team_colors[2] = model.cluster_centers_[1]

    def get_player_team(self, frame, bbox, pid):
        if pid in self.player_team_dict:
            return self.player_team_dict[pid]
        color = self.get_player_color(frame, bbox)
        team = self.kmeans.predict(color.reshape(1, -1))[0] + 1
        self.player_team_dict[pid] = team
        return team

# ----------------------------- BALL ASSIGNMENT MODULE --------------------------------

class PlayerBallAssigner:
    def __init__(self):
        self.max_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        ball_pos = get_center_of_bbox(ball_bbox)
        closest = float('inf')
        assigned = -1
        for pid, player in players.items():
            box = player['bbox']
            d1 = measure_distance((box[0], box[3]), ball_pos)
            d2 = measure_distance((box[2], box[3]), ball_pos)
            dist = min(d1, d2)
            if dist < self.max_distance and dist < closest:
                closest = dist
                assigned = pid
        return assigned

# ----------------------------- CAMERA MOVEMENT MODULE --------------------------------

class CameraMovementEstimator:
    def __init__(self, frame):
        self.min_shift = 5
        self.lk_params = dict(winSize=(15,15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        mask[:,0:20] = 1
        mask[:,900:1050] = 1
        self.features = dict(maxCorners=100, qualityLevel=0.3, minDistance=3, blockSize=7, mask=mask)

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        shifts = [[0, 0]] * len(frames)
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_pts = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for i in range(1, len(frames)):
            gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            new_pts, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, old_pts, None, **self.lk_params)

            max_dist, dx, dy = 0, 0, 0
            for new, old in zip(new_pts, old_pts):
                new_p, old_p = new.ravel(), old.ravel()
                dist = measure_distance(new_p, old_p)
                if dist > max_dist:
                    max_dist = dist
                    dx, dy = measure_xy_distance(old_p, new_p)

            if max_dist > self.min_shift:
                shifts[i] = [dx, dy]
                old_pts = cv2.goodFeaturesToTrack(gray, **self.features)
            old_gray = gray.copy()

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(shifts, f)

        return shifts

    def add_adjust_positions_to_tracks(self, tracks, shifts):
        for obj, obj_tracks in tracks.items():
            for i, track in enumerate(obj_tracks):
                for tid, info in track.items():
                    pos = info['position']
                    dx, dy = shifts[i]
                    info['position_adjusted'] = (pos[0] - dx, pos[1] - dy)

    def draw_camera_movement(self, frames, shifts):
        output = []
        for i, frame in enumerate(frames):
            draw = frame.copy()
            overlay = draw.copy()
            cv2.rectangle(overlay, (0,0), (500, 100), (255,255,255), -1)
            cv2.addWeighted(overlay, 0.6, draw, 0.4, 0, draw)
            dx, dy = shifts[i]
            cv2.putText(draw, f"Camera Movement X: {dx:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
            cv2.putText(draw, f"Camera Movement Y: {dy:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
            output.append(draw)
        return output

# ----------------------------- VIEW TRANSFORMER MODULE --------------------------------

class ViewTransformer:
    def __init__(self, source_points=None, target_points=None):
        if source_points is not None and target_points is not None:
            if isinstance(source_points, list):
                source_points = np.array(source_points, dtype=np.float32)
            if isinstance(target_points, list):
                target_points = np.array(target_points, dtype=np.float32)
                
            if source_points.shape != target_points.shape:
                raise ValueError("Source and target must have the same shape.")
            if source_points.shape[1] != 2:
                raise ValueError("Source and target points must be 2D coordinates.")

            self.homography_matrix, _ = cv2.findHomography(source_points, target_points)
            if self.homography_matrix is None:
                raise ValueError("Homography matrix could not be calculated.")
        else:
            self.homography_matrix = None
        
        # Pitch dimensions
        self.pitch_dims = (400, 300)
        self.pitch_real_dims = (105, 68)

    def transform_point(self, point):
        if self.homography_matrix is None or point is None:
            return point
            
        point = np.array([point], dtype=np.float32).reshape(-1, 1, 2)
        transformed_point = cv2.perspectiveTransform(point, self.homography_matrix)
        return transformed_point.reshape(-1, 2)[0]

    def transform_points(self, points):
        if self.homography_matrix is None or len(points) == 0:
            return points
            
        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(points, self.homography_matrix)
        return transformed_points.reshape(-1, 2)

    def create_2d_pitch_view(self, tracks, frame_idx):
        pitch = np.zeros((self.pitch_dims[1], self.pitch_dims[0], 3), dtype=np.uint8)
        
        # Background
        pitch[:, :] = (50, 180, 50)
        
        # Pitch lines
        cv2.rectangle(pitch, (10, 10), (self.pitch_dims[0]-10, self.pitch_dims[1]-10), (255, 255, 255), 1)
        cv2.line(pitch, (self.pitch_dims[0]//2, 10), (self.pitch_dims[0]//2, self.pitch_dims[1]-10), (255, 255, 255), 1)
        cv2.circle(pitch, (self.pitch_dims[0]//2, self.pitch_dims[1]//2), 20, (255, 255, 255), 1)
        
        # Scale factors
        scale_x = (self.pitch_dims[0] - 20) / self.pitch_real_dims[0]
        scale_y = (self.pitch_dims[1] - 20) / self.pitch_real_dims[1]
        
        # Draw players and ball
        for obj_type in tracks:
            if frame_idx >= len(tracks[obj_type]):
                continue
                
            frame_tracks = tracks[obj_type][frame_idx]
            for track_id, info in frame_tracks.items():
                if 'transformed_position' not in info:
                    continue
                    
                pos = info['transformed_position']
                x = int(10 + pos[0] * scale_x)
                y = int(10 + pos[1] * scale_y)
                
                if x < 0 or x >= self.pitch_dims[0] or y < 0 or y >= self.pitch_dims[1]:
                    continue
                
                if obj_type == 'ball':
                    color = (0, 255, 255)
                    radius = 3
                else:
                    color = info.get('team_color', (0, 0, 255))
                    radius = 5
                
                cv2.circle(pitch, (x, y), radius, color, -1)
                if obj_type == 'players':
                    cv2.putText(pitch, str(track_id), (x-5, y-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return pitch

    def add_transformed_position_to_tracks(self, tracks):
        for obj_type in tracks:
            for frame_tracks in tracks[obj_type]:
                for track_id, info in frame_tracks.items():
                    if 'position' in info:
                        info['transformed_position'] = self.transform_point(info['position'])

    def draw_2d_view_on_frame(self, frame, tracks, frame_idx):
        pitch_view = self.create_2d_pitch_view(tracks, frame_idx)
        
        # Position at bottom center
        x_offset = (frame.shape[1] - self.pitch_dims[0]) // 2
        y_offset = frame.shape[0] - self.pitch_dims[1] - 10
        
        roi = frame[y_offset:y_offset + self.pitch_dims[1],
                   x_offset:x_offset + self.pitch_dims[0]]
        
        cv2.addWeighted(pitch_view, 0.5, roi, 0.3, 0, roi)
        frame[y_offset:y_offset + self.pitch_dims[1],
              x_offset:x_offset + self.pitch_dims[0]] = roi
        
        return frame

# ----------------------------- SPEED AND DISTANCE MODULE --------------------------------

class SpeedAndDistance_Estimator:
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24
        # Store position history for each player
        self.position_history = {}
        # Store total distance for each player
        self.total_distance = {}
        # Store previous frame positions
        self.prev_positions = {}

    def update_speed_and_distance(self, frame_tracks):
        """
        Update speed and distance for players in real-time
        """
        current_positions = {}
        
        # Process players
        if 'players' in frame_tracks:
            for tid, info in frame_tracks['players'].items():
                if 'transformed_position' not in info:
                    continue
                    
                current_pos = info['transformed_position']
                current_positions[tid] = current_pos
                
                # Initialize history if needed
                if tid not in self.position_history:
                    self.position_history[tid] = []
                    self.total_distance[tid] = 0
                
                # Add current position to history
                self.position_history[tid].append(current_pos)
                
                # Keep only the last frame_window positions
                if len(self.position_history[tid]) > self.frame_window:
                    self.position_history[tid].pop(0)
                
                # Calculate speed if we have enough history
                if len(self.position_history[tid]) >= 2:
                    # Get previous position from history
                    prev_pos = self.position_history[tid][-2]
                    
                    # Calculate distance
                    d = measure_distance(prev_pos, current_pos)
                    self.total_distance[tid] += d
                    
                    # Calculate speed (km/h)
                    t = 1 / self.frame_rate  # Time between frames
                    speed = (d / t) * 3.6  # Convert to km/h
                    
                    # Add speed and distance to player info
                    info['speed'] = speed
                    info['distance'] = self.total_distance[tid]
        
        # Update previous positions
        self.prev_positions = current_positions

    def draw_speed_and_distance(self, frame, frame_tracks):
        """
        Draw speed and distance on the frame
        """
        if 'players' in frame_tracks:
            for tid, info in frame_tracks['players'].items():
                if 'speed' in info and 'bbox' in info:
                    pos = get_foot_position(info['bbox'])
                    pos = (int(pos[0]), int(pos[1] + 40))
                    speed = info.get('speed')
                    dist = info.get('distance')
                    cv2.putText(frame, f"{speed:.2f} km/h", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                    cv2.putText(frame, f"{dist:.2f} m", (pos[0], pos[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        return frame

    # Keep the original method for batch processing if needed
    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        for obj, obj_tracks in tracks.items():
            if obj in ('ball', 'referees'): continue
            for i in range(0, len(obj_tracks), self.frame_window):
                j = min(i + self.frame_window, len(obj_tracks) - 1)
                for tid in obj_tracks[i]:
                    if tid not in obj_tracks[j]: continue
                    start = obj_tracks[i][tid].get('position_transformed')
                    end = obj_tracks[j][tid].get('position_transformed')
                    if not start or not end: continue
                    d = measure_distance(start, end)
                    t = (j - i) / self.frame_rate
                    speed = (d / t) * 3.6
                    total_distance.setdefault(obj, {}).setdefault(tid, 0)
                    total_distance[obj][tid] += d
                    for k in range(i, j):
                        if tid in obj_tracks[k]:
                            obj_tracks[k][tid]['speed'] = speed
                            obj_tracks[k][tid]['distance'] = total_distance[obj][tid]

# ----------------------------- MAIN FUNCTION --------------------------------

def main():
    video_path = "invd/test1.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    print("Initializing models...")
    tracker = Tracker("models/best.pt")
    team_assigner = TeamAssigner()
    ball_assigner = PlayerBallAssigner()
    speed_distance_estimator = SpeedAndDistance_Estimator()
    
    # 2D projection points
    source_points = np.array([
        [100, 200],
        [1820, 200],
        [0, 900],
        [1920, 900]
    ], dtype=np.float32)
    
    target_points = np.array([
        [0, 0],
        [105, 0],
        [0, 68],
        [105, 68]
    ], dtype=np.float32)
    
    view_transformer = ViewTransformer(source_points, target_points)

    tracks = {
        "players": [],
        "referees": [],
        "ball": []
    }
    
    team_control = []
    frame_count = 0
    first_frame_processed = False
    
    print("Processing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get detections
        detections = tracker.model.predict(frame, conf=0.1)[0]
        
        # Initialize frame tracks
        frame_tracks = {
            "players": {},
            "referees": {},
            "ball": {}
        }
        
        # Process detections
        cls_map = detections.names
        rev_map = {v: k for k, v in cls_map.items()}
        det_sv = sv.Detections.from_ultralytics(detections)
        
        # Update goalkeeper labels
        for j, cid in enumerate(det_sv.class_id):
            if cls_map[cid] == "goalkeeper":
                det_sv.class_id[j] = rev_map['player']
        
        # Get tracking results
        result = tracker.tracker.update_with_detections(det_sv)
        
        # Process tracking results
        for d in result:
            bbox, cid, tid = d[0].tolist(), d[3], d[4]
            if cid == rev_map['player']:
                frame_tracks['players'][tid] = {'bbox': bbox}
            elif cid == rev_map['referee']:
                frame_tracks['referees'][tid] = {'bbox': bbox}
        
        # Process ball detections
        for d in det_sv:
            bbox, cid = d[0].tolist(), d[3]
            if cid == rev_map['ball']:
                frame_tracks['ball'][1] = {'bbox': bbox}

        # Initialize team colors
        if not first_frame_processed and frame_tracks['players']:
            team_assigner.assign_team_color(frame, frame_tracks['players'])
            first_frame_processed = True

        # Assign team colors
        if first_frame_processed:
            for pid, player in frame_tracks['players'].items():
                team = team_assigner.get_player_team(frame, player['bbox'], pid)
                player['team_color'] = team_assigner.team_colors[team]
        
        # Add positions
        for obj_type in frame_tracks:
            for tid, info in frame_tracks[obj_type].items():
                bbox = info['bbox']
                pos = get_center_of_bbox(bbox) if obj_type == 'ball' else get_foot_position(bbox)
                info['position'] = pos
                info['transformed_position'] = view_transformer.transform_point(pos)
        
        # Update speed and distance in real-time
        speed_distance_estimator.update_speed_and_distance(frame_tracks)
        
        # Assign ball possession
        if frame_tracks['ball']:
            ball_bbox = frame_tracks['ball'][1]['bbox']
            possession = ball_assigner.assign_ball_to_player(frame_tracks['players'], ball_bbox)
            if possession is not None and possession >= 0:
                if possession in frame_tracks['players']:
                    frame_tracks['players'][possession]['has_ball'] = True
                    team_control.append(team_assigner.player_team_dict.get(possession, 0))
                else:
                    team_control.append(0)
            else:
                team_control.append(0)
        else:
            team_control.append(0)
        
        # Draw annotations
        annotated_frame = frame.copy()
        
        # Draw players
        for pid, player in frame_tracks['players'].items():
            color = player.get('team_color', (0, 0, 255))
            annotated_frame = tracker._draw_ellipse(annotated_frame, player['bbox'], color, pid)
            if player.get('has_ball', False):
                annotated_frame = tracker._draw_triangle(annotated_frame, player['bbox'], (0, 0, 255))
        
        # Draw referees
        for ref in frame_tracks['referees'].values():
            annotated_frame = tracker._draw_ellipse(annotated_frame, ref['bbox'], (0, 255, 255))
        
        # Draw ball
        for ball in frame_tracks['ball'].values():
            annotated_frame = tracker._draw_triangle(annotated_frame, ball['bbox'], (0, 255, 0))
        
        # Draw speed and distance
        annotated_frame = speed_distance_estimator.draw_speed_and_distance(annotated_frame, frame_tracks)
        
        # Draw possession bar
        annotated_frame = tracker._draw_possession_bar(annotated_frame, frame_count, team_control)
        
        # Add 2D view
        annotated_frame = view_transformer.draw_2d_view_on_frame(annotated_frame, {'players': [frame_tracks['players']], 
                                                                                 'referees': [frame_tracks['referees']], 
                                                                                 'ball': [frame_tracks['ball']]}, 0)
        
        # Show progress
        print(f"\rProcessing frame {frame_count+1}", end="")
        
        # Display frame
        cv2.imshow('Football Analysis', annotated_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        # Update tracks
        tracks['players'].append(frame_tracks['players'])
        tracks['referees'].append(frame_tracks['referees'])
        tracks['ball'].append(frame_tracks['ball'])
        
        frame_count += 1
    
    print(f"\nProcessed {frame_count} frames")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
