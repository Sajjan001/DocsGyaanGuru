from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

def load_document(path, extension):
    loaders = {
        "pdf": PyPDFLoader,
        "txt": TextLoader,
        "docx": Docx2txtLoader
    }

    if extension not in loaders:
        raise ValueError("Unsupported file type")
    
    loader = loaders[extension](path)
    return loader.load()
