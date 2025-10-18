# Pneumonia Detection System

A medical imaging analysis tool that uses deep learning to detect pneumonia from chest X-rays.

## Overview

This system analyzes chest X-ray images and provides risk assessments based on clinical symptoms. Built during my MSc in AI, it combines computer vision with practical healthcare applications.

## What It Does

- Analyzes chest X-rays for pneumonia detection
- Calculates patient risk scores based on symptoms and medical history
- Generates detailed medical reports with visualizations
- Provides confidence scores for predictions

## Technical Details

**Model Performance:**
- Architecture: ResNet-18 CNN
- Training accuracy: 85.58%
- Dataset: Chest X-Ray Images (Pneumonia)

**Built With:**
- Python 3.12
- PyTorch for deep learning
- Streamlit for the web interface
- ReportLab for PDF generation
- Matplotlib for visualizations

## Project Structure
```
pneumonia-detection-system/
├── app/
│   └── pneumonia_detector.py
├── models/
│   └── best_model.pth
├── requirements.txt
└── README.md
```

## Running Locally

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `streamlit run app/pneumonia_detector.py`

## Current Limitations

- Requires manual review by medical professionals
- Not validated for clinical use
- Model trained on limited dataset

## Future Improvements

- Expand training dataset
- Add explainability features (Grad-CAM)
- Implement uncertainty quantification
- Compare against published benchmarks

## Author

Ayoolumi Melehon
MSc in Artificial Intelligence

## Disclaimer

This is a research and educational project. Not intended for clinical diagnosis. All results should be validated by qualified healthcare professionals.
