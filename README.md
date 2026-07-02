# Simple Photo Editor

A small Streamlit app for editing photos using OpenCV. Upload an image, adjust it with
sliders and checkboxes, and download the result.

## Description

This project was built as a hands-on exercise to practice image processing with OpenCV
and building a UI with Streamlit. It covers the basics — resizing, brightness/contrast,
filters — plus a couple of extra effects like a portrait-style background blur and a
cartoon filter.

## Tools Used

- Python
- Streamlit
- OpenCV
- NumPy
- Pillow

## Features

- Upload an image (jpg / jpeg / png)
- Resize using a scale slider
- Adjust brightness and contrast
- Convert to grayscale
- Blur effect
- Warm filter (boosts red, reduces blue)
- Sharpen effect
- Portrait-style background blur (detects a face and blurs the background)
- Bonus effects: edge detection, sketch effect, cartoon effect, rotation
- Download the edited image as PNG

## How to Run

1. Clone this repo:
   ```
   git clone <your-repo-url>
   cd photo-editor
   ```

2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the app:
   ```
   streamlit run app.py
   ```

4. Open the link Streamlit gives you in the terminal (usually `http://localhost:8501`).

## Notes

- The portrait blur uses OpenCV's built-in Haar cascade face detector. If no face is
  found, it just keeps the center of the image sharp instead.
- All effects can be combined at the same time using the sidebar controls.
