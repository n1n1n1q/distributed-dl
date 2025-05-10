import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
path1 = os.path.join("models", "resnet18_mnist_1.pth")
path2 = os.path.join("models", "resnet18_mnist_2.pth")

@st.cache_resource
def load_models():
    model1 = torch.load(path1, map_location="cpu") if os.path.exists(path1) else None
    model2 = torch.load(path2, map_location="cpu") if os.path.exists(path2) else None
    return model1, model2

model1, model2 = load_models()

st.title("distributed dl demo")
st.write(
    "Draw a digit (0–9) on the left canvas.\n"
    "A resnet with random weights and a resnet with trained weights will predict the digit."
)

draw_col, preview_col = st.columns(2)

with draw_col:
    st.subheader("Draw here")
    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 0)", stroke_width=10,
        stroke_color="#000000", background_color="#FFFFFF",
        height=280, width=280, drawing_mode="freedraw", key="canvas"
    )
    if canvas.image_data is not None:
        arr = canvas.image_data.astype(np.uint8)
        img = Image.fromarray(arr).convert("L")
        st.session_state.img = img

with preview_col:
    st.subheader("Preview and predictions")
    if 'img' in st.session_state:
        img_gray = st.session_state.img
        st.image(img_gray, caption="input", width=280)

        preprocess = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda t: t.repeat(3, 1, 1))
        ])
        inp = preprocess(img_gray).unsqueeze(0)

        with torch.no_grad():
            out1, out2 = model1(inp), model2(inp)
        pred1 = out1.argmax(1).item()
        pred2 = out2.argmax(1).item()

        st.write(f"**Random model prediction:** {pred1}")
        st.write(f"**Trained model prediction:** {pred2}")
