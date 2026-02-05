# MindTrack-AI-Attention-Monitoring
Real-time AI-based attention monitoring system using computer vision and eye behavior analysis.
🧠 MindTrack – AI-Based Real-Time Attention Monitoring System

MindTrack is a real-time, non-intrusive attention monitoring system that uses computer vision and AI to analyze eye behavior and head movement through a webcam. It classifies a person’s attention state as Focused, Distracted, or Sleepy without using any physical sensors or datasets.

📌 Features

🎥 Real-time webcam-based monitoring
👁️ Eye Aspect Ratio (EAR)–based sleepiness detection
🔄 Head movement–based distraction detection
🚨 Alert system for loss of attention
📊 Simple and easy-to-understand visual reports
📝 Grouped attention logs (easy Excel analysis)
❌ No dataset or training required

🎯 Main Objective

To automatically detect and analyze human attention levels in real time using eye behavior and computer vision techniques in a low-cost, non-intrusive manner.

🏫 Applications

Classroom and online education monitoring
Student engagement analysis
Driver drowsiness detection
Online exam proctoring
Workplace productivity monitoring

🤖 AI & Technology Used

Artificial Intelligence
MediaPipe FaceMesh (pre-trained deep learning model by Google)
Rule-based AI logic for attention classification

⚠️ No dataset is required because the facial landmark model is already trained.

🛠️ Tech Stack
Technology	Purpose
Python	Core programming language
OpenCV	Webcam access and video processing
MediaPipe	Face and eye landmark detection
NumPy	Mathematical calculations
Matplotlib	Graph and report generation
CSV	Attention log storage

🧠 How It Works

Webcam captures live video frames
MediaPipe detects facial landmarks
Eye landmarks are extracted
Eye Aspect Ratio (EAR) is calculated
Head movement is analyzed
Rule-based logic determines attention state
Logs and visual reports are generated

📊 Output

Real-time attention state displayed on screen
Attention duration logs (grouped by time intervals)
Simple graphs understandable by non-technical users
