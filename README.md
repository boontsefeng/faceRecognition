# To Download Brew on Mac:
/bin/bash -c "$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh](https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh))"

# To Download Python with Brew
brew install python

# After downloading Python, go into the folder that you want the project to be in

# Create Virtual Environment
python3 -m venv .venv

# Activate Virtual Environment
source .venv/bin/activate

# Download Mediapipe (Facial Recognition library)
pip install opencv-python mediapipe

# For code modification, make sure to add the photos under the same folder as the .py itself

# Example of activating the code:
cd ~/Desktop/pose-recognizer
source .venv/bin/activate
python pose_recognizer.py

# A window will pop up showing your camera feed and a pose window, it will display the photos according to the pose
