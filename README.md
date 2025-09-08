# Soccerlytics-Advanced-Football-Video-Analysis-System
DESCRIPTION:
Soccerlytics is an comprehensive football video analysis system that tracks and analyzes matches. It generates enhanced visualizations with player tracking, ball possession statistics, team assignments, and a unique 2D top-down view for tactical analysis.

## 🌟 Key Features

-   *🔎 Player & Ball Tracking: Utilizes a state-of-the-art **YOLO* model for high-accuracy detection of players, referees, and the ball. This is combined with the *ByteTrack* algorithm to assign unique IDs and track their movements seamlessly throughout the video.

-   *🎨 Automatic Team Assignment: The system intelligently identifies team affiliations by analyzing player jersey colors. It uses **K-Means clustering* on the pixel data from each player's torso to automatically group players into two distinct teams and color-code them.

-   *⚽ Real-time Ball Possession: FootRec calculates ball possession by determining the closest player to the ball in every frame. This data is used to update a dynamic **possession bar* on the screen, showing the percentage of control for each team as the match progresses.

-   *🗺 2D Tactical View (Bird's-Eye View): A standout feature of the system. Using a **homography transformation*, it maps the players' and ball's positions from the camera perspective onto a miniature 2D representation of the football pitch, offering a clear, tactical overview of formations and movements.

-   *🎥 Camera Movement Compensation: To ensure positional data is accurate, the system employs the **Lucas-Kanade optical flow* method. This estimates the camera's pan and tilt, allowing the system to differentiate between player movement and camera movement for more precise tracking.

-   *⚡ Performance Metrics (Speed & Distance): The system estimates the **speed (in km/h)* and *total distance covered (in meters)* for each player. These crucial performance statistics are calculated from the players' transformed positions and displayed in real-time.

-   *💾 Data Caching for Efficiency*: Computationally intensive tasks like object tracking and camera movement estimation can be saved to pickle files (stubs). This allows for rapid re-analysis and debugging without needing to re-process the entire video from scratch.

---

## 💻 Tech Stack

-   *Python*
-   *OpenCV* for video processing and computer vision tasks.
-   *Ultralytics YOLO* for object detection.
-   *Supervision* for tracking and annotation utilities.
-   *NumPy* & *Pandas* for numerical operations and data handling.
-   *Scikit-learn* for K-Means clustering.
-   *Matplotlib* for visualization.

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

-   Python 3.9+
-   Git
-   A trained YOLO model file (e.g., best.pt).
-   A video of a football match (e.g., test1.mp4).

### Installation

1.  *Clone the repository:*
    bash
    git clone [https://github.com/your-username/FootRec.git](https://github.com/your-username/FootRec.git)
    cd FootRec
    

2.  *Create a virtual environment (recommended):*
    bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    

3.  *Install the required dependencies:*
    *(You may need to create a requirements.txt file based on the imports in the script)*
    bash
    pip install opencv-python numpy pandas ultralytics supervision scikit-learn matplotlib
    

4.  *Set up project structure:*
    -   Create a models/ directory and place your trained best.pt file inside it.
    -   Create an invd/ directory and place your input video file (e.g., test1.mp4) inside it.

### Usage

To run the analysis, simply execute the main script from the root directory:

```bash
python main_final.py 
