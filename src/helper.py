from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

#Extract Data From the PDF
def load_pdf_file(data):
    loader=DirectoryLoader(data,
                          glob="*.pdf",
                          loader_cls=PyPDFLoader)
    
    documents =  loader.load()

    return documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    text_chunks = []
    for doc in extracted_data:
        text_chunks.extend(text_splitter.split_text(doc.page_content))  # Extracting text correctly

    return text_chunks



#Download the Embedding from Hugging Face
def download_hugging_face_embedding():
    # Download the embedding from Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings