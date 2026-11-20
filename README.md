
# CFU Colony Annotation Tool

## Project Overview
The CFU Colony Annotation Tool is a Python-based application designed to detect and annotate bacterial colonies in Petri dish images.

It uses YOLOv8 for automatic colony detection and Streamlit to provide an interactive user interface. It also supports manual annotation.

## Technologies Used
- Python
- Streamlit
- YOLOv8
- Ultralytics
- OpenCV
- Pillow
- Streamlit Drawable Canvas

## Features
- Upload Petri dish images
- Automatic colony detection using YOLOv8
- Display bounding boxes around detected colonies
- Show total detection count
- Display class-wise colony counts
- Display prediction confidence
- Export YOLO predictions as JSON
- Manually annotate colonies using rectangles
- Export manual annotations as JSON

## Model Classes
1. B.subtilis
2. C.albicans
3. Contamination
4. E.coli
5. P.aeruginosa
6. S.aureus

## Project Structure

```text
CFU_Manual_Annotation_App/
│
├── app.py
├── requirements.txt
├── README.md
│
└── models/
    └── best (2).pt
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project folder:

```bash
cd CFU_Manual_Annotation_App
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Application

```bash
python -m streamlit run app.py
```

## Important Note
The model provides prototype predictions. Detection results should be independently validated before using them for scientific or laboratory reporting.
