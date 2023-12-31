import streamlit as st
import streamlit.components.v1 as components
from analysis_pipeline import AnalysisPipeline

st.set_page_config(page_title="Visualize Language Similarity", layout="wide")
st.title("Visualize Language Similarity")


try:
    pipeline = st.session_state['pipeline']
    
    st.markdown("""
    This stage of analysis is optional, but can be helpful in certain situations--particularly when looking for
    memetic content or the prevelance of specific talking points.

    Unfortunately, network visualizations are highly processor-intensive to render, and due to the limited resources available on this community server, this stage of analysis will not run without crashing. Instead, I've added a pre-rendered version of the network below with 12 topics.

    If you would like to run this analysis yourself, please download this repository and deploy locally in Docker.""")


    def create_network(minimum_similarity):
        pipeline.minimum_similarity = minimum_similarity
        with st.spinner("Running Similarity Analysis"):
            pipeline.calculate_similarity()
        with st.spinner("Rendering Network. This may take some time."):
            pipeline.create_network().generate_visualization()
        return pipeline

    def create_for_download(minimium_similarity):
        pipeline.minimum_similarity = minimum_similarity
        with st.spinner("Running Similarity Analysis"):
            pipeline.calculate_similarity()
        with st.spinner("Rendering Network. This may take some time."):
            pipeline.create_network()
        with st.spinner("Preparing file for download"):
            pipeline.save_network()
        return pipeline

    
    with st.form("run network"):
        col1, col2, col3 = st.columns(3)
        with col1:
            submitted_1 = st.form_submit_button("Visualize Random Sample")
        with col2:
            submitted_2 = st.form_submit_button("Prep Full Dataset for Download")
        with col3:
            with st.expander("Advanced Options"):
                st.markdown("The slider below sets the minimum similarity between two documents required for them to be 'linked' in the network. To cut down on processing time, I have limited these options to relatively high similarity, however different situations call for different settings.")
                minimum_similarity = st.slider("Minimum Similarity for Linkage", min_value = 0.3, max_value =0.99, value = 0.45)

    if submitted_1:
        #create_network(minimum_similarity)
        #HtmlFile = open('graph.html', 'r', encoding = 'utf-8')
        #source_code = HtmlFile.read()
        #components.html(source_code, height=800, width=800)
        st.write("This feature is not available on Streamlit Cloud, please deploy this repo locally in Docker.")
    if submitted_2:
        #create_for_download(minimum_similarity)
        #with open("similarity_analysis.gexf", 'r') as file:
        #    st.download_button("Download Results", data=file, file_name = 'similarity_results.gexf')
        st.write("This feature is not available on Streamlit Cloud, please deploy this repo locally in Docker.")

    st.markdown("""Example network using 12 topics""")
    htmlFile = open('app/example_graph.html', 'r', encoding='utf-8')
    source_code=htmlFile.read()
    components.html(source_code, height=800, width=1000)

    st.session_state['pipeline'] = pipeline
    
except:
    st.markdown("""The topic model has not been run yet! Please return to the "Run Topic Model" page to proceed with the similarity analysis.""")