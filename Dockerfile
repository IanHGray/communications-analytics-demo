
FROM python:3.9
WORKDIR /app
COPY ./requirements.txt requirements.txt
RUN apt update && \
    apt-get install pkg-config && \
    python -m venv /py && \
    /py/bin/pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python -m nltk.downloader stopwords && \
    python -m nltk.downloader wordnet && \
    python -m nltk.downloader omw-1.4 && \
    python -m nltk.downloader punkt && \
    python -m nltk.downloader vader_lexicon

EXPOSE 8501
COPY . /app
CMD streamlit run app/streamlit_app.py