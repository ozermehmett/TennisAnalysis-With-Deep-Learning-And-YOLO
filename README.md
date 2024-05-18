# Tennis Analysis with Deep Learning and YOLO

## Overview
The **Tennis Analysis with Deep Learning and YOLO** project is a comprehensive solution designed for analyzing tennis matches using
state-of-the-art deep learning techniques. Leveraging the power of YOLO (You Only Look Once) 
object detection, this project offers robust player and ball detection capabilities.
Additionally, it utilizes Convolutional Neural Networks (CNNs) to extract court 
key points for advanced analysis. Whether you're a tennis enthusiast, a data scientist, or a machine learning practitioner, 
this project provides an immersive learning experience in computer vision and sports analytics.

## Features
- **Player and Ball Detection:** Utilizes YOLO for accurate detection of tennis players and the ball in video frames.
- **Speed Measurement:** Estimates the speed of players and the shot speed of the ball, providing valuable insights into player performance.
- **Shot Counting:** Counts the number of shots made by each player, facilitating detailed performance analysis.
- **Court Key points:** Extracts key points from the tennis court, enabling advanced spatial analysis and visualization.

## Repository Contents
   - The repository contains various components, including:
     - `constants`: Constants used in the project.
     - `court_line_detector`: Detection of court lines.
     - `input_videos`: Input video files.
     - `models`: Model-related files.
     - `output_videos`: Output video files.
     - `runs`: Detection and prediction runs.
     - `tracker_stubs`: Tracker stubs.
     - `trackers`: Tracking components.
     - `training`: Training-related files.
     - `utils`: Utility functions.
     - `Main.py`: Main script for the project.
     - `cuda_installl`: CUDA installation information.
     - `requirements.txt`: Required Python packages.
     - `tennis_court_dimensions.png`: Tennis court dimensions reference.
     - `yolo_interface.py`: Interface for YOLO-based detection⁵.


## Getting Started
Follow these steps to set up and run the project:

1. **Clone the Repository:**
   ```
   git clone https://github.com/ozermehmett/TennisAnalysis-With-Deep-Learning-And-YOLO.git
   cd TennisAnalysis-With-Deep-Learning-And-YOLO
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.6 or higher installed. Then, install the required packages:
   ```
   pip install -r requirements.txt
   ```
3.  **Installing PyTorch Version**
   To ensure compatibility with the project, it's recommended to install a specific version of PyTorch. You can install PyTorch using the following command:
   ```
   pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Run the Analysis:**
   Execute the main script to analyze a tennis video:
   ```
   python Main.py --input_video input_videos/tennis_match.mp4
   ```

5. **View Results:**
   Once the analysis is complete, the processed video with player and ball detections will be saved in the `output_videos` directory.
## Sample Output
 Watch the sample output video to see the project in action:

https://github.com/ozermehmett/TennisAnalysis-With-Deep-Learning-And-YOLO/assets/115498182/f50b4124-36a5-4447-abef-d67666a971ab


## Additional Notes
- **Tennis Court Dimensions:** Refer to `tennis_court_dimensions.png` for detailed tennis court dimensions to enhance your analysis.
- **Exploring the Code:** Delve into various modules such as `court_line_detector`, `trackers`, and `utils` to gain insights into the implementation details and customize the project according to your requirements.

Feel free to explore and expand upon the project to uncover deeper insights into tennis analytics and player performance. Whether you're conducting research, building applications, or simply indulging in your passion for tennis, this project offers endless possibilities for exploration and innovation.
