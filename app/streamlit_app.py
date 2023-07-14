import streamlit as st
import streamlit.components.v1 as components
from analysis_pipeline import AnalysisPipeline
from PIL import Image

st.set_page_config(page_title="Communications Analytics", layout="wide")
st.title("A Beginner's Guide to Communications Analytics")

image = Image.open('app/network_example.png')
st.image(image)
st.markdown("""This example application walks through the analysis performed in this Medium post, and allows for the underlying data source and settings to be tweaked.

To begin, click on the *Run Topic Model* page on the left, and proceed through the following steps.

While this is by no means a comprehensive approach to analyzing social and traditional media conversations, I believe it is a useful (and practical) starting point.
For more information about the code used to build this application, please check out [this repository on GitHub](https://github.com/IanHGray/communications-analytics-demo), or find me at [ianhgray.github.io](https://ianhgray.github.io).
""")


#Example dataset can be found at: https://www.kaggle.com/datasets/rmisra/news-category-dataset

