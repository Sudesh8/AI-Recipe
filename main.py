import streamlit as st
import PIL
from PIL import Image
import numpy as np

from utils import get_ingredients, getMyrecipe


uploadedIMG = st.file_uploader("Choose a file", type=["png", "jpeg", "jpg"])
cusine_option = st.selectbox(
    "Select your cusine...",
    ("Indian", "Pakistani", "French"),
)


st.write("Selected Cusine is :", cusine_option)

btn = st.button("Start The Process")

# if uploadedIMG is not None:
if btn:
    st.image(uploadedIMG)
    pilimage = Image.open(uploadedIMG).convert("RGB")
    imgArray = np.asarray(pilimage)

    predIngredients = list(get_ingredients(IMG_arr=imgArray))

    AiRecipe = getMyrecipe(myIngredients=predIngredients, mycuisine=cusine_option)

    print(AiRecipe)

    st.write(AiRecipe)
