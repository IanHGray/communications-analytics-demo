import json
import pandas as pd
import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
from pyvis import network as net
from color_map import color_map
import random
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.lda_model
import plotly.graph_objects as go
import colorlover as cl

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('vader_lexicon')

sns.set(style='whitegrid', context='talk')
lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))
punctuation = set(string.punctuation)
add_punctuation = ['“', '‘', '‘', '’', '`', '``', '‘‘'] #There are some punctuation quirks in the example dataset. Opted to pull them out individually rather than regex.
punctuation.update(add_punctuation)
sid = SentimentIntensityAnalyzer()

class AnalysisPipeline:
    def __init__(self):
        self.source_dataframe = None
        self.example_corpus = 'app/News_Category_Dataset_v3.json'
        self.example_category = "POLITICS"
        self.num_clusters = None
        self.additional_stopwords = []
        self.notebook = False
        self.buttons = False
        self.minimum_similarity = 0.5
        self.remove_isolated_nodes = True
        self.random_sample = True
        self.random_sample_num = 1000
        self.num_top_words = 20
        self.max_df = 0.90
        self.min_df = 0.01
    
    def load_corpus(self): 
        self.corpus = self.source_dataframe.content.tolist()
        return self

    def load_example_corpus(self):
        source_data = []
        f = open(self.example_corpus)
        for item in f:
            raw_data = json.loads(item)
            if raw_data['category'] == self.example_category:
                data_row = {}
                data_row['content'] = raw_data['short_description']
                data_row['uuid'] = raw_data['link'] #Because this dataset does not have unique identifiers, we will substitute a uuid with a unique url
                data_row['title'] = raw_data['headline']
                source_data.append(data_row)
        self.source_dataframe = pd.DataFrame(source_data)
        corpus = self.source_dataframe.content.tolist()
        self.corpus = [i.lower() for i in corpus]  
        return self

    def process_text(self):
        stopwords.update(self.additional_stopwords)
        self.cleaned_corpus = []
        for text in self.corpus:
            tokenized = word_tokenize(text)
            lowered = [i.lower() for i in tokenized]
            punctuation_removed = [i for i in lowered if i not in punctuation]
            stopwords_removed = [i for i in punctuation_removed if i not in stopwords]
            lemmatized = [lemmatizer.lemmatize(word) for word in stopwords_removed]
            self.cleaned_corpus.append(lemmatized)
        return self

    def create_tfidf(self):
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False, max_df=self.max_df, min_df=self.min_df)
        self.cv_vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False, max_df=self.max_df, min_df=self.min_df)
        self.tfidf_array = self.tfidf_vectorizer.fit_transform(self.cleaned_corpus)
        self.count_vector_array = self.cv_vectorizer.fit_transform(self.cleaned_corpus)
        self.vocab_tfidf = self.tfidf_vectorizer.get_feature_names_out()
        self.vocab_cv = self.cv_vectorizer.get_feature_names_out()
        return self

    def run_lda(self):
        if self.num_clusters is None:
            print("Number of clusters not set. Please add argument")
        else:
            self.lda_model = LatentDirichletAllocation(n_components = self.num_clusters,
                                                    max_iter = 20,
                                                    random_state = 42)
            self.X_topics = self.lda_model.fit_transform(self.tfidf_array)
            self.topic_words = self.lda_model.components_
            return self

    def print_topics(self):
        for i, topic_dist in enumerate(self.topic_words):
            sorted_topic_dist = np.argsort(topic_dist)
            topic_words = np.array(self.vocab_tfidf)[sorted_topic_dist]
            topic_words = topic_words[:-self.num_top_words:-1]
            print("Topic:", str(i+1), topic_words)
        return self

    def assign_topics(self):
        doc_topic = self.lda_model.transform(self.tfidf_array)
        topic_list = []
        for n in range(doc_topic.shape[0]):
            topic_doc = doc_topic[n].argmax()
            topic_list.append(topic_doc)
        document_ids = self.source_dataframe['uuid'].tolist()
        if len(document_ids) == len(topic_list):
            zipped = zip(document_ids, topic_list)
            topic_df = pd.DataFrame(zipped, columns=['uuid', 'topic'])
            self.annotated_dataframe = pd.merge(self.source_dataframe, topic_df, on='uuid')
            self.annotated_dataframe['topic'] = self.annotated_dataframe['topic'].astype(str) #Converts the topic number to a string. This is important for creating the network map later.
            return self
        else:
            print("Length mismatch between source dataframe and LDA output. Check pre-processing")

    def calculate_similarity(self):
        similarity = self.tfidf_array * self.tfidf_array.T
        self.similarity = similarity.multiply(similarity >= self.minimum_similarity)
        return self


    def create_network(self):
        self.annotated_dataframe['color'] = self.annotated_dataframe['topic'].map(color_map) #Gets the hexidecimal code associated with each topic number.   
        attributes_dict = {}
        for i in range(0, len(self.annotated_dataframe)):
            attr_row = {}
            attr_row['uuid'] = self.annotated_dataframe['uuid'][i]
            attr_row['topic'] = self.annotated_dataframe['topic'][i]
            attr_row['label'] = self.annotated_dataframe['title'][i]
            attr_row['color'] = self.annotated_dataframe['color'][i]
            attributes_dict[i] = attr_row
            
        self.G = nx.from_scipy_sparse_matrix(self.similarity)
        if self.random_sample == True:
            random_sample_edges = random.sample(list(self.G.edges), self.random_sample_num)
            self.G = nx.Graph()
            self.G.add_edges_from(random_sample_edges)
        
        nx.set_node_attributes(self.G, values=attributes_dict)
        self.G.remove_edges_from(nx.selfloop_edges(self.G)) #Removes self-loops in the network, where a node is connected to itself.
        self.G.remove_nodes_from(list(nx.isolates(self.G)))
        return self

    def save_network(self):
        nx.write_gexf(self.G, 'similarity_analysis.gexf')
        return self
        
    def generate_visualization(self):
        #This segment draws heavily from the gist below:
        #https://gist.github.com/quadrismegistus/92a7fba479fc1e7d2661909d19d4ae7e
        pyvis_graph = net.Network(
            height="600px",
            width="100%",
            bgcolor="#FFFFFF",
            font_color="black",    
            notebook=self.notebook,
            directed=False,
            )
        for node, node_attrs in self.G.nodes(data=True):
            pyvis_graph.add_node(str(node),
                                 color = node_attrs['color'],
                                 label = node_attrs['label'])
        for source, target, edge_attrs in self.G.edges(data=True):
            if not 'value' in edge_attrs and not 'width' in edge_attrs and 'weight' in edge_attrs:
                edge_attrs['value']=edge_attrs['weight']
            pyvis_graph.add_edge(str(source), str(target), **edge_attrs)
        if self.buttons == True:
            pyvis_graph.width = '75%'
            pyvis_graph.show_buttons()
        if self.notebook == True:
            pyvis_graph.width='75%'
        pyvis_graph.save_graph('graph.html')
        return self

    def generate_wordclouds(self):
        fig = plt.figure(figsize=(15,12))
        fig.subplots_adjust(hspace=0.5)
        fig.suptitle("Key Words by Topic")
        for i in range(self.num_clusters):
            ax = plt.subplot(math.ceil(self.num_clusters/2), 2, i+1)
            topic = 'Topic ' + str(i)
            text = ' '.join(self.annotated_dataframe.loc[self.annotated_dataframe['topic']==str(i), 'content'].values)
            wordcloud = WordCloud(width=1000, height=800, random_state=42, background_color='white', collocations=False).generate(text)
            ax.imshow(wordcloud) 
            ax.set_title(topic)
            ax.axis("off")
        self.fig = fig
        return self
    
    def generate_lda_viz(self):
        self.lda_viz = pyLDAvis.lda_model.prepare(self.lda_model, self.count_vector_array, self.cv_vectorizer)
        self.lda_viz_html = pyLDAvis.prepared_data_to_html(self.lda_viz)
        return self

    def calculate_vader_sentiment(self):
        def run_vader(content):
            ss = sid.polarity_scores(content)
            sentiment = ss['compound']
            return sentiment
        def categorize_sentiment(vader_sentiment):
            if vader_sentiment >= 0.35:
                return "positive"
            elif vader_sentiment <= -0.35:
                return "negative"
            else:
                return "neutral" 

        self.annotated_dataframe['vader_sentiment'] = list(map(run_vader, self.annotated_dataframe['content']))
        self.annotated_dataframe['vader_category'] = list(map(categorize_sentiment, self.annotated_dataframe['vader_sentiment']))
        return self

    def generate_sentiment_viz(self):
        def format_topic_str(topic):
            topic = int(topic) + 1 #Removes zero ordering
            return "Topic " + str(topic)
        self.annotated_dataframe['topic_str'] = list(map(format_topic_str, self.annotated_dataframe['topic']))
        pivoted = pd.pivot_table(
            self.annotated_dataframe, 
            index='topic_str',
            columns ='vader_category',
            values = 'uuid',
            aggfunc = 'count')
        category_order = ['negative', 'neutral', 'positive']
        pivoted = pivoted[category_order]
        pivoted.negative = pivoted.negative * -1
        pivoted = pivoted.sort_values(by = 'negative', ascending = False)

        fig = go.Figure()
        for column in pivoted.columns:
            fig.add_trace(go.Bar(
                x = pivoted[column],
                y = pivoted.index,
                name = column,
                orientation = 'h',
                marker_color = cl.scales[str(len(category_order))]['div']['RdYlGn'][category_order.index(column)],
            ))
        fig.update_layout(
            barmode = 'relative',
            title='Sentiment Breakdown by Topic'
            )
        self.sentiment_fig = fig

        return self