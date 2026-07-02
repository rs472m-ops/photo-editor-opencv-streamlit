import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(page_title="Simple Photo Editor", layout="wide")

st.title("📷 Simple Photo Editor")
st.write("Upload a photo, play with the sliders on the left, and download the result.")

# ---------- helper functions ----------

def adjust_brightness_contrast(img, brightness, contrast):
    # brightness is just added to every pixel, contrast scales it
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


def apply_blur(img, amount):
    if amount == 0:
        return img
    k = amount * 2 + 1  # kernel size has to be odd
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_sharpen(img):
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def apply_warm_filter(img):
    img = img.astype(np.int16)
    img[:, :, 2] = np.clip(img[:, :, 2] + 25, 0, 255)  # more red
    img[:, :, 0] = np.clip(img[:, :, 0] - 20, 0, 255)  # less blue
    return img.astype(np.uint8)


def portrait_blur(img):
    # blur the whole photo, then use a face detector to keep the face area sharp
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    blurred = cv2.GaussianBlur(img, (35, 35), 0)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    if len(faces) == 0:
        # no face found, just keep the center of the image sharp
        h, w = img.shape[:2]
        cv2.ellipse(mask, (w // 2, h // 2), (w // 4, h // 3), 0, 0, 360, 255, -1)
    else:
        for (x, y, w, h) in faces:
            cx, cy = x + w // 2, y + h // 2
            cv2.ellipse(mask, (cx, cy), (int(w * 0.9), int(h * 1.3)), 0, 0, 360, 255, -1)

    mask = cv2.GaussianBlur(mask, (31, 31), 0)  # soften the edges of the mask
    mask_3ch = cv2.merge([mask, mask, mask]) / 255.0
    result = (img * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
    return result


def apply_edge_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def apply_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def apply_cartoon(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)


# ---------- sidebar controls ----------

uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    st.sidebar.header("Resize")
    scale = st.sidebar.slider("Scale (%)", 10, 200, 100)
    new_w = int(img.shape[1] * scale / 100)
    new_h = int(img.shape[0] * scale / 100)
    img = cv2.resize(img, (new_w, new_h))

    st.sidebar.header("Brightness & Contrast")
    brightness = st.sidebar.slider("Brightness", -100, 100, 0)
    contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)
    img = adjust_brightness_contrast(img, brightness, contrast)

    st.sidebar.header("Filters")
    grayscale = st.sidebar.checkbox("Grayscale")
    blur_amount = st.sidebar.slider("Blur", 0, 15, 0)
    warm = st.sidebar.checkbox("Warm filter")
    sharpen = st.sidebar.checkbox("Sharpen")
    portrait = st.sidebar.checkbox("Portrait background blur")

    st.sidebar.header("Extra effects")
    edge = st.sidebar.checkbox("Edge detection")
    sketch = st.sidebar.checkbox("Sketch effect")
    cartoon = st.sidebar.checkbox("Cartoon effect")
    rotate_angle = st.sidebar.slider("Rotate", -180, 180, 0)

    # apply everything in order
    if warm:
        img = apply_warm_filter(img)
    if blur_amount > 0:
        img = apply_blur(img, blur_amount)
    if sharpen:
        img = apply_sharpen(img)
    if portrait:
        img = portrait_blur(img)
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if edge:
        img = apply_edge_detection(img)
    if sketch:
        img = apply_sketch(img)
    if cartoon:
        img = apply_cartoon(img)
    if rotate_angle != 0:
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), rotate_angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h))

    # ---------- show before / after ----------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(pil_img, use_container_width=True)
    with col2:
        st.subheader("Edited")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

    # ---------- download button ----------
    result_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    st.download_button(
        label="Download edited image",
        data=buf.getvalue(),
        file_name="edited_image.png",
        mime="image/png",
    )

else:
    st.info("Upload an image from the sidebar to get started.")
