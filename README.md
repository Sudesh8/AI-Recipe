
# AI-Recipe

This project demonstrates a recipe generation system that combines object detection with and natural language processing using LangChain and HuggingFace models. The system detects ingredients from an image and generates a recipe based on the detected ingredients and user-specified cuisine

## Project Overview

- The application provides the following functionality:

1. Ingredient Detection: Uses a YOLO model to detect ingredients from uploaded images.

2. Recipe Generation: Uses a language model to generate a recipe based on detected ingredients and the chosen cuisine.

3. User Interface: Built with Streamlit for easy interaction and visualization.
## Requirements

- Python 3.x
- ultralytics
- opencv-python-headless
- langchain
- transformers
- streamlit
- Pill (for image handling)
- numpy

  ### How to install requirements
`
pip install -r requirements.txt
`
### How to run source code
`
python utils.py
`
## Usage

- Run the Streamlit App:

- Interact with the App:
1. Upload an image containing ingredients.
2. Select your desired cuisine.
3. Click "Start The Process" to view the generated recipe based on detected ingredients.


