# app/analyzer.py
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd



# Cache the model to improve performance and reduce reloading
def load_sentence_transformer_model():
    return SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

class IncidentAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """

        Initialize the analyzer with a DataFrame containing incident data.
        This will load the pre-trained Sentence Transformer model and encode
        the incident descriptions to create embeddings.

        :param df: The DataFrame containing the incident data.
        """
        self.df = df
        self.model = load_sentence_transformer_model()
        
        # Create embeddings for all incident descriptions using the model
        self.embeddings = self.model.encode(df["combined_text"].tolist(), convert_to_numpy=True)
        
        # Create a FAISS index to store the embeddings and quickly search for similar ones
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def retrieve_similar(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """
        Retrieve the top-k most similar incidents to the query using cosine similarity.

        :param query: The query text (e.g., user input).
        :param top_k: The number of similar incidents to retrieve.
        :return: A DataFrame containing the top-k similar incidents.
        """
        # Encode the query into a vector (embedding)
        query
