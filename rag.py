from langchain.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader("")
PDF_data = loader.load()