import streamlit as st
import numpy as np
import streamlit as st
import pandas as pd
import contractions
import plotly.graph_objects as go
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from io import BytesIO
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from pyvis.network import Network
import streamlit.components.v1 as components
import spacy
import matplotlib.pyplot as plt


#How to figure out which page is being run like url_scraper the button was clicked or main.py

st.set_page_config(page_title="Dataviz", page_icon="📈")

st.markdown("# Data Visualization")
st.sidebar.header("Data Visualization")
st.sidebar.write("This section focuses on interpreting the results of the tiktok/youtube videos")

# Load the English language model
nlp = spacy.load("en_core_web_sm")

def remove_stopwords(text):
    # Get spaCy's set of English stopwords
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    # Tokenize the text using spaCy
    doc = nlp(text)
    # Filter out stopwords
    filtered_tokens = [token.text for token in doc if token.text.lower() not in stop_words]
    return " ".join(filtered_tokens)

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

print(st.session_state.page_state)

# Initialize data variable
data = None


if st.session_state.page_state == "main":
    # Retrieve data from session state
    data = st.session_state.get("data", None)
    print(data)
    print("hello1")
elif st.session_state.page_state == "url_scraper":
    # Retrieve dataurl from session state
    data = st.session_state.get("dataurl", None)
    print(data)
    print("hello2")

print(data)

st.subheader("This section examines the video transcripts and generates visual representations to enhance our comprehension")
#Printing overall bar chart showing the number of misinformation vs no of non misinformation videos
st.text("Misinformation Status Frequency for Videos Uploaded")
misinformation_count = 0
non_misinformation_count = 0
for video in data:
    if video['misinformation_status'] == 'Misinformation detected':
        misinformation_count += 1
    else:
        non_misinformation_count += 1


chart_data = {
    'Category': ['Misinformation Detected', 'No Misinformation Detected'],
    'Count': [misinformation_count, non_misinformation_count]
}

fig, ax = plt.subplots()
ax.bar(chart_data['Category'], chart_data['Count'])

# Set y-axis ticks to integer values
ax.set_yticks(range(max(chart_data['Count']) + 1))

# Add labels and title
ax.set_xlabel('Category')
ax.set_ylabel('Count')
ax.set_title('Misinformation Detection')

# Show the plot
st.pyplot(fig)

    
for l in range(0,len(data)):
    #expanding the contractions in the text
    contracted_data=contractions.fix(data[l]["transcript"])
    text_data = remove_stopwords(contracted_data)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text_data])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    top_keywords = [feature_names[idx] for idx in tfidf_matrix.toarray()[0].argsort()[-10:][::-1]]
    relationships = defaultdict(int)
    tokens = text_data.lower().split()
    for i in range(len(tokens)):
        if tokens[i] in top_keywords:
            for j in range(i + 1, len(tokens)):
                if tokens[j] in top_keywords:
                    relationships[(tokens[i], tokens[j])] += 1
    nodes = set()
    edges = []
    for relationship, weight in relationships.items():
        source, target = relationship
        if source != target:  
            nodes.add(source)
            nodes.add(target)
            edges.append((source, target, weight))
    st.write("Figure: Network depicting the relationship between the Top 10 Frequent Words for "+str(data[l]["video_file"]))
    net = Network(height="500px", width="100%")
    for node in nodes:
        net.add_node(node, label=node, shape="dot")
    max_weight = max([weight for _, _, weight in edges])
    for source, target, weight in edges:
        scaled_weight = 1 + 2 * (weight / max_weight)
        net.add_edge(source, target, value=scaled_weight, width=scaled_weight*0.2)
    net.barnes_hut(gravity=-1000, central_gravity=0.3, spring_length=200)
    path = '/tmp'
    net.save_graph(f'{path}/pyvis_graph.html')
    HtmlFile = open(f'{path}/pyvis_graph.html','r',encoding='utf-8')
    components.html(HtmlFile.read(), height=435)
import os
def remove_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted {file_path}")
            elif os.path.isdir(file_path):
                remove_files(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# Define the directory path where the files are stored (e.g., 'tmp' directory)
directory_path = "tmp"

# Remove all files in the specified directory
remove_files(directory_path)