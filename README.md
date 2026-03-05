# Vision-Based Navigation Assistant for Low-Vision Users

A computer vision system that provides **actionable navigation cues** to visually impaired users by analyzing street scenes.  
The system combines **object detection**, **monocular depth estimation**, and **spatial reasoning** to generate **natural language guidance** such as:

> "Car near ahead. Motorcycle on your left."

The project focuses on **cluttered Indian street environments**, which contain unique objects like **autorickshaws, riders, motorcycles, and animals on the road**.

---

# Project Objective

This project builds a system that answers those questions using:

1. **Object Detection**
2. **Depth Estimation**
3. **Spatial Reasoning**
4. **Navigation Cue Generation**

---

# System Architecture

The navigation system processes an image using the following pipeline:
```
Input Image
│
├── Object Detection (YOLOv8)
│ Detect vehicles, people, animals, etc.
│
├── Depth Estimation (MiDaS)
│ Predict relative distance for every pixel
│
├── Spatial Reasoning
│ Determine object direction and proximity
│
└── Navigation Cue Generation
    Generate spoken instructions
```

---

# Implementation

## 1. Object Detection

Object detection is performed using **YOLOv8 (You Only Look Once)**.

### Model Used
- `YOLOv8n` (default lightweight model)
- Later fine-tuned on **IDD (Indian Driving Dataset)**

### YOLOv8 Architecture
YOLOv8 consists of:

1. **Backbone**
   - CSPDarknet-style convolution layers
   - Extracts hierarchical image features

2. **Neck**
   - PAN-FPN feature pyramid
   - Combines multi-scale features

3. **Detection Head**
   - Anchor-free object detection
   - Predicts bounding box, class, confidence

### Training Dataset
Default model is trained on **COCO dataset**

COCO contains:
- 118k training images
- 80 object categories

Examples:
```
person, car, truck, bus, motorcycle, bicycle
```

### Detection Output
Each detected object contains:

```
{
bbox: [x1, y1, x2, y2],
label: "car",
confidence: 0.89
}
```

---

# 2. Depth Estimation

Depth estimation uses **MiDaS**, a monocular depth estimation model.

Unlike stereo cameras, MiDaS estimates **depth from a single image**.

### Model Used
```
dpt_levit_224
```

This model uses:

- Vision Transformer backbone
- Lightweight architecture
- ~51 million parameters

### Training Data

MiDaS is trained on **multiple datasets combined**, including:

- ReDWeb
- MegaDepth
- DIW
- TartanAir
- KITTI

This multi-dataset training allows strong generalization.

### Output

MiDaS outputs a **depth map**:

```
bright pixels → close objects
dark pixels → far objects
```

Example:

```
depth_map[y, x] = relative depth value
```

---

# 3. Spatial Reasoning

The spatial reasoning module converts detections into **navigational information**.

For each detected object:

### Direction Estimation

Based on horizontal position:

```
far-left   if center_x < image_width * 0.2
left       if image_width * 0.2 < center_x < image_width * 0.4
center     if middle region
right      if image_width * 0.6 < center_x < image_width *0.8
far-right  if image_width * 0.8 < center_x
```

### Distance Estimation

Using depth map values:

```
very close
near
moderate distance
far
```

### Risk Score

Objects are prioritized using:

```
risk_score = a * depth_score + b * lateral_score
```

Objects closer to the **center walking corridor** receive higher priority.

---

# 4. Navigation Cue Generation

The system converts prioritized hazards into **spoken instructions**.

Example:

```
car near ahead
motorcycle on your left
```

The system limits output to the **top 2 hazards** to avoid overwhelming the user.

---

# Project Novelty

Most vision systems provide **scene captions**.

This project instead generates **actionable navigation cues**.

Key innovations:

### 1. Vision-to-Navigation pipeline
Transforms detection outputs into **movement guidance**.

### 2. Depth-aware risk prioritization
Uses **3D depth information** instead of only 2D bounding boxes.

### 3. Hazard prioritization
Focuses on **relevant obstacles** rather than all detected objects.

### 4. Indian street adaptation
Supports classes like:

```
autorickshaw
rider
motorcycle
animal
```

which are often missing from standard datasets.

---

# Project Structure

```
vlm-navigation-assistant
│
├── models
│   ├── detector.py
│   ├── spatial_reasoning.py
│   ├── navigation_agent.py
│   ├── scene_graph.py
│   └── depth_estimator.py
│
├── caption
│   └── rule_based_caption.py
│
├── utils
│   └── visualize.py
│
├── tts
│   └── speak.py
│
├── samples
│   └── road.jpg
│
├── main.py
├── requirements.txt
└── README.md
```

---

# How to Run the Project

## 1. Clone Repository

```bash
git clone https://github.com/<your-username>/vlm-navigation-assistant.git
cd vlm-navigation-assistant
```

## 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 4. Install MiDaS

Clone the MiDaS repository:
```bash
git clone https://github.com/isl-org/MiDaS.git
```

Download model weights:
```
https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_levit_224.pt
```
Place file inside:
```
MiDaS/weights/
```

## 5. Run the Navigation System
```bash
python main.py --image samples/road.jpg
```
Example output:
```
Generated: car near ahead. motorcycle on your left.
```
The system will also save a visualization:
```
outputs/road_boxed.jpg
```

# Future Improvements

## Planned upgrades:

Video input instead of images

Multi-frame input and dynamic frame stitcher

Lane-aware navigation

Real-world distance estimation

Fine-tuned model on Indian Road Dataset (IDD)

Enhanced visualization and Spatial reasoning
