import streamlit as st
from PIL import Image

st.set_page_config(page_title="About Me", layout="wide")
st.title("About Me")

image = Image.open('app/scotland.jpeg')
st.image(image, width = 600)

st.markdown("""
I am a Houston, Texas-based data scientist and communications strategist specializing in network analysis and mapping media ecosystems. Over the past decade, I have served in a variety of political, consulting, and defense contracting roles--all focused on applying analytics solutions to seemingly intractible problems.

I currently serve as Head of Data Science for US Strategic Communications at FTI Consulting, and have the privilege of leading a talented and energetic group of data scientists and engineers with varied experience in network analysis, natural language processing, and artificial intelligence.

Our team has tackled a wide range of uniquely complex client needs, including developing a robust disinformation monitoring and mitigation system, establishing new techniques to identify thought leaders and communities of influence, and leveraging automation and AI techniques to speed time-to-insight in crisis situations. We take an integrated, cross-segment approach to problem-solving and always welcome fresh challenges.

For more information about the team and our work, please [visit our website](http://fticommunications.com) or feel free to [contact me directly](https://ianhgray.github.io/#contact).

""")