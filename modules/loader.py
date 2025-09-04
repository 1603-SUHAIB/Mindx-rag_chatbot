from PyPDF2 import PdfReader

def load_document(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return ""
