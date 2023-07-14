import streamlit as st
import streamlit.components.v1 as components
from analysis_pipeline import AnalysisPipeline

st.set_page_config(page_title="Run LDA Topic Model", layout="wide")
st.title("Run LDA Topic Model")

st.markdown("""
We begin the analysis process by running an LDA topic model. This allows us to separate the overall conversation into distinct themes. To begin, select a desired number of topics to divide the conversation into. (I usually start with a relatively small number and work my way up.)

""")

def generate_example(num_clusters, min_df, max_df, example_category, additional_stopwords):
    pipeline = AnalysisPipeline()
    pipeline.num_clusters = num_clusters
    pipeline.min_df = min_df
    pipeline.max_df = max_df
    pipeline.example_category = example_category
    pipeline.additional_stopwords = additional_stopwords
    with st.spinner("Loading and Pre-Processing Dataset"):
        pipeline.load_example_corpus().process_text().create_tfidf()
        num_records = len(pipeline.corpus)
    with st.spinner(f"Running Topic Model on {str(num_records)} Records"):
        pipeline.run_lda().assign_topics()
    with st.spinner("Visualizing LDA Results"):
        pipeline.generate_lda_viz()
    return pipeline

with st.form("run analysis"):
    num_clusters = st.number_input('Desired Number of Topics', min_value = 2, max_value = 12, value = 4)
    with st.expander("Advanced Options"):
        st.markdown("Our example analysis was run on articles with a 'Political' tag, but the model will work similarily with other categories.")
        example_category = st.selectbox("Choose News Category", ('Politics', 'World News', 'Entertainment', 'Parenting'))
        st.markdown("""
            These parameters set the minimum and maximum percentage of posts a given word can appear in before it is excluded. 
        
            For instance, in a politically-oriented conversation, the words "Donald Trump" are not analytically useful, as they will appear across most articles, and provide no additional information beyond being the subject of discussion. By removing these highly common words, we can more easily hone into the content that helps us draw distinctions.
                """)
        col1, col2 = st.columns(2)
        with col1:
            min_df = st.slider("Minimum Term Appearence (%)", min_value = 0.0, max_value = 1.0, value = 0.01)
        with col2:
            max_df = st.slider("Maximum Term Appearence (%)", min_value = 0.0, max_value = 1.0, value = 0.90)
        st.markdown("""
        'Stopwords' are common english words (such as prepositions, pronouns, and common verbs) that are necessary for our (human) understanding of a text, but are not useful for analysis.
        
        If you find an unhelpful word appearing as prominent within the analysis, try adding it here to remove it and make room for more useful terms.
        """)

        additional_stopwords = st.text_area("Add Additional Stopwords Here (Comma Separated)")
    submitted = st.form_submit_button("Run Analysis")

if submitted:
    example_category = example_category.upper()
    if additional_stopwords is not None:
        additional_stopwords = additional_stopwords.split(',')
    else:
        additional_stopwords = []
    pipeline = generate_example(num_clusters, min_df, max_df, example_category, additional_stopwords)
    
    st.markdown("""
            The plots generated below are a two-dimensional representation of the model, with each circle being a separate topic. Their relative closeless represents how semantically similar they are. When they overlap, it indicates that the model had a hard time differentiating between their themes.
            What we're looking for is good separation between the topics, avoiding overlap to the extent possible.
            """)
    components.html(pipeline.lda_viz_html, height=1000, width=1200)

    if 'pipeline' not in st.session_state:
        st.session_state['pipeline'] = pipeline