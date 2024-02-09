from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


def load_data(path):
    loader = DirectoryLoader(path, glob="**/*.xml")
    docs = loader.load()
    return docs

def embedding(model_name, model_kwargs, encode_kwargs):
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embedding_model

def find_data(docs, data_name):
    for doc in docs:
        if doc.metadata['source'].find(data_name) != -1:
            return doc.page_content
    
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)