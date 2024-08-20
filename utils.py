### yolo
### LLM
import ultralytics
import cv2
import langchain, transformers, langchain_community
from ultralytics import YOLO
import numpy as np
from langchain import HuggingFaceHub, LLMChain
from langchain_core.prompts import PromptTemplate
import os
from pprint import pprint


# Load the YOLO model trained on a custom dataset (veggie.pt)
model = YOLO("veggie.pt")

# Define model names for text generation
Mname = "google/flan-t5-base"
zmodel = "HuggingFaceH4/zephyr-7b-beta"
mistral = "mistralai/Mistral-Nemo-Instruct-2407"

# Define configuration for the language model
config = {
    "temperature": 0.1,
    "max_length": 7024,
    "max_new_tokens": 1024,
    "return_full_text": False,
}


# CONSTANTS
# Set the HuggingFace API token as an environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load the language model from HuggingFaceHub
llm = HuggingFaceHub(repo_id=mistral, model_kwargs=config)


# Function to get ingredients from the image
# IMG = cv2.imread("img.jpg")


def get_ingredients(IMG_arr):
    result = model.predict(IMG_arr)
    indexList = result[0].boxes.cls.tolist()
    indexList = np.array(indexList).astype(int)
    ClassName = list((result[0].names.values()))

    predIngredients = []
    for i in indexList:
        # print(i)
        predIngredients.append(ClassName[i])
        # print(pred)

    return set(predIngredients)


mytemp = """
you are a chef bot.you give recipies for provided ingredients list.
for the given ingredients list write a recipe.
in return just give recipe and recipe name.

ingredients: {ingredients}
cuisine: {cuisine}

recipe:
"""


prompt = PromptTemplate.from_template(mytemp)

# Create an LLMChain object with the prompt and the language model
chain = LLMChain(prompt=prompt, llm=llm)


# predIngredients = list(get_ingredients(IMG_arr=IMG))


def getMyrecipe(myIngredients, mycuisine):
    # Generate the recipe based on the list of ingredients
    # ans = chain.run({"ingredients": myIngredients})
    ans = chain.run({"ingredients": myIngredients, "cuisine": mycuisine})

    return ans


# myrecp = getMyrecipe(myIngredients=predIngredients)

# print("___________________________________")
# pprint(myrecp)
# print(type(myrecp))
# print("___________________________________")

print("ENDEDDD")
