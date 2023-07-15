import streamlit as st
from analysis_pipeline import AnalysisPipeline

st.set_page_config(page_title="Assess Sentiment", layout="wide")
st.title("Assess Sentiment")

def generate_sentiment_visualization():
    with st.spinner("Generating Sentiment Breakdown"):
        pipeline.calculate_vader_sentiment().generate_sentiment_viz()
        return pipeline


try:
    pipeline = st.session_state['pipeline']

    st.write("With the topics defined, we can now explore the tone and intensity of language used in each topic. This analysis uses the [Vader Lexicon](https://medium.com/@piocalderon/vader-sentiment-analysis-explained-f1c4f9101cd9), a widely used model for media analysis.")
    with st.form("Generate Sentiment Breakdown"):
        submitted = st.form_submit_button("Generate Sentiment Breakdown")

    if submitted:
        pipeline = generate_sentiment_visualization()
        st.plotly_chart(pipeline.sentiment_fig)

    st.session_state['pipeline'] = pipeline

except:
    st.markdown("""The topic model has not been run yet! Please return to the "Run Topic Model" page to proceed with the sentiment analysis.""")