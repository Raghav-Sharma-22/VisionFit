# VISION-FIT

An AI-powered workout assistant using OpenCV and MediaPipe to track exercise reps, calories, and time in real-time.

## Features
- **Exercise Tracking:** Automated rep counting for Push-Ups and Jumping Jacks.
- **Live Metrics:** Real-time calorie estimation and workout timer.
- **Visual Feedback:** Interactive movement bars and skeletal landmark overlay.
- **Session Summary:** Post-workout report with total reps, time, and calories.

## Prerequisites
```bash
pip install opencv-python mediapipe numpy
```

## Controls
| Key | Action |
| :--- | :--- |
| **P** | Select Push-Ups |
| **J** | Select Jumping Jacks |
| **S** | Start Workout |
| **Q** | End Session / Exit |
| **R** | Return to Menu / Reset |
| **ESC** | Exit Application |

## How to Run
1. Ensure your webcam is connected.
2. Run the script:
   ```bash
   python vision_fit.py
   ```
3. Stand back so your full body is visible to the camera for best accuracy.
