import PyPDF2
import os

def parse_pdf(file_path):
    """Parse a PDF file and extract text content."""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
    except Exception as e:
        print(f"Error parsing {file_path}: {str(e)}")
        return None

def parse_pdfs_in_directory(input_dir):
    """Parse all PDF files in a directory and return a list of (filename, text) tuples."""
    pdf_texts = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(input_dir, filename)
            text = parse_pdf(file_path)
            if text:
                pdf_texts.append((filename, text))
    return pdf_texts

if __name__ == "__main__":
    # Example usage
    input_directory = "./app/data"
    pdf_texts = parse_pdfs_in_directory(input_directory)
    for filename, text in pdf_texts:
        print(f"Parsed {filename}: {text[:100]}...")