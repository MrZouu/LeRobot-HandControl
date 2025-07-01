# LeRobot-HandControl

<p> 
Hand gesture‑based control interface for robotics systems. This project was built during the Hugging Face Hackathon 2025, enabling robots to react to live hand gestures using real‑time calibration and detection. 
</p> 
<br/>
<p align="center">
	<img src="assets/Hand_Control_Video.gif" width="700">
</p>

<br/>
⚙️ Hugging Face Hackathon 2025 — created within the event timeframe, focused on gesture‑based robot control.

<br/>

# Summary

* **[Summary](#summary)**
* **[Dependencies](#dependencies)**
* **[Getting Started](#getting-started)**
* **[Credits](#credits)**

# Dependencies

* [**Python**](https://www.python.org/)
* [**MediaPipe 0.10.5**](https://pypi.org/project/mediapipe/) – Cross-platform framework for building multimodal applied ML pipelines
* [**protobuf 4.21.12**](https://pypi.org/project/protobuf/) – Protocol Buffers serialization library by Google
* [**pyserial 3.5**](https://pypi.org/project/pyserial/) – Python library for serial communication
* [**OpenCV 4.7.0.68**](https://opencv.org/) – Computer vision library

# Getting Started
### Hand Gesture Recognition Module

This module uses **MediaPipe** to detect and track 3D hand landmarks in real time using a standard webcam. It is part of the HuggingFace LeRobot project and intended to be run on with any LeRobot product.

---

## 1. Clone the Repository
```bash
git clone https://github.com/yourusername/bluexplorer.git
cd bluexplorer/gesture-control
```

## 2. Create and Activate a Virtual Environment (optional but recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies
```bash
pip install mediapipe==0.10.5 protobuf==4.21.12 pyserial==3.5 opencv-python==4.7.0.68
```

## 4. Run the calibration
For calibration, use a checkerboard with 11.5 cm squares and take at least 10 captures, or adapt the code for different dimensions.
<p>
	<img src="./assets/pattern.png" width="30%">
</p>

```bash
python3 camera_calibration.py
```

## 5. Run the Gesture Recognition
```bash
python3 detect_hand.py
```

#  Credits
* [**Lorenzo**](https://github.com/MrZouu) : Creator of the project.
