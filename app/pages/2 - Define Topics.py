import streamlit as st
from analysis_pipeline import AnalysisPipeline

st.set_page_config(page_title="Define Topics", layout="wide")
st.title("Define Topics")

try:
    pipeline = st.session_state['pipeline']

    st.markdown("""
    Now that we have established and run our LDA model, we can interpret the results and assign labels to them for further analysis.

    Click the button below to generate a series of word clouds, each displaying the most representative words for their respective topics.

    This process is often iterative. If results are too vague, consider increasing the number of topics on the previous page and rerunning the model. Similarily, if they are too specific, try increasing the number of topics.
    """)

    def generate_visualization():
        with st.spinner("Generating Word Clouds"):
            pipeline.generate_wordclouds()
        return pipeline

    with st.form("Generate Word Clouds"):
        submitted = st.form_submit_button("Generate Word Clouds")

    if submitted:
        pipeline = generate_visualization()
        st.pyplot(fig=pipeline.fig)

        #HtmlFile = open('graph.html', 'r', encoding = 'utf-8')
        #source_code = HtmlFile.read()
        #components.html(pipeline.lda_viz_html, height=800, width=800)

    st.session_state['pipeline'] = pipeline

    st.write("##")
except:
    st.markdown("""The topic model has not been run yet! Please return to the "Run Topic Model" page to proceed with the model breakdown""")