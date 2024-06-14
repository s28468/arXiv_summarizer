import warnings

warnings.filterwarnings('ignore', category=FutureWarning, message=".*`resume_download` is deprecated.*")

import io
import os
import re

import fitz
import requests
import torch
import argparse
from PIL import Image
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from llmsherpa.readers import LayoutPDFReader
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Spacer, Image as ReportLabImage, Paragraph
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from mongo_server import MongoDBSaver


def download_pdf(pdf_url, output_path='arXiv_doc.pdf'):
    """
    Download a PDF from the specified URL and save it to a local file.

    Args:
        pdf_url (str): The URL of the PDF to download.
        output_path (str, optional): The local file path to save the downloaded PDF. Defaults to 'arXiv_doc.pdf'.

    Returns:
        str: The path to the downloaded PDF file.
    """
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(output_path, 'wb') as file:
            file.write(response.content)
    return output_path


def extract_text_from_pdf(pdf_url):
    """
    Extract text from a PDF at the specified URL.

    Args:
        pdf_url (str): The URL of the PDF from which to extract text.

    Returns:
        str: The extracted text from the PDF.

    Example:
        >>> extract_text_from_pdf("arXiv_doc.pdf")
        'Extracted text from the PDF...'
    """
    loader = PyPDFLoader(pdf_url)
    return str(loader.load())


def extract_headers_from_html(doc):
    """
    Extract headers from an HTML document and return them as a list.

    This function identifies headers in the HTML document, processes them to ensure unique entries,
    and checks for the presence of an "Introduction" header. It also processes headers to include
    "REFERENCES" if it is found in the document text.

    Args:
        doc (str): The HTML document as a string.

    Returns:
        list: A list of processed header texts.
    """
    soup = BeautifulSoup(doc, 'html.parser')
    headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'td'])
    headers_text = [header.get_text(strip=True).strip() for header in headers]
    headers_text = list(dict.fromkeys(headers_text))

    pattern = re.compile(r'^\d+(\.\d+)*\s')
    numbered_headers = [header for header in headers_text if pattern.match(header)]
    intro_present = any(re.match(r'^1\s+Introduction', header, re.IGNORECASE) for header in numbered_headers)
    if not intro_present:
        index_for_intro = next((i for i, header in enumerate(numbered_headers) if header.startswith('2 ')), 0)
        numbered_headers.insert(index_for_intro, '1 Introduction')

    updated_pattern = re.compile(r'''
        ^(\d+(\.\d+)*)(\s+[A-Za-z].*)$|^\d+\s\d+\.\d+\s(\d+\.\d+\s+)*\d+K?\.\d+(\s\d+)*$
    ''', re.VERBOSE)
    headers = [header for header in numbered_headers if updated_pattern.match(header)]
    headers.append("REFERENCES" if "REFERENCES" in doc else "References")

    return headers


def extract_images_from_pdf(pdf_path, images_folder='images'):
    """
    Extract images from a PDF and save them to a specified folder.

    This function opens a PDF file, extracts all images, and saves them in the specified folder
    in PNG format. It creates the folder if it does not exist.

    Args:
        pdf_path (str): The path to the PDF file.
        images_folder (str, optional): The folder to save the extracted images. Defaults to 'images'.

    Returns:
        int: The number of images extracted.
    """
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    doc = fitz.open(pdf_path)
    photo_index = 0

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            imageFileName = os.path.join(images_folder, f'image_{photo_index}.png')
            image.save(imageFileName, format='PNG')
            photo_index += 1

    doc.close()
    return photo_index


def summarize_text(text, tokenizer, model):
    """
    Summarize a long text using a pre-trained transformer model.

    This function tokenizes the input text, processes it in manageable chunks,
    generates summaries for each chunk using the specified model, and combines the
    summaries into a single summarized text.

    Args:
        text (str): The input text to be summarized.
        tokenizer: The tokenizer to preprocess the input text.
        model: The pre-trained model to generate the summaries.

    Returns:
        str: The summarized text.
    """
    inputs_no_trunc = tokenizer(text, max_length=None, return_tensors='pt', truncation=False)

    chunk_start = 0
    chunk_end = 1024
    inputs_batch_lst = []
    space_token_id = tokenizer.encode(' ', add_special_tokens=False)[0]

    while chunk_start <= len(inputs_no_trunc['input_ids'][0]):
        try:
            current_chunk = inputs_no_trunc['input_ids'][0][chunk_start:chunk_end].tolist()
            end_index = len(current_chunk) - 1 - current_chunk[::-1].index(space_token_id)
            chunk_end = chunk_start + end_index
        except ValueError:
            pass

        inputs_batch = inputs_no_trunc['input_ids'][0][chunk_start:chunk_end]
        inputs_batch = torch.unsqueeze(inputs_batch, 0)
        inputs_batch_lst.append(inputs_batch)
        chunk_start = chunk_end + 1
        chunk_end = min(chunk_start + 1024, len(inputs_no_trunc['input_ids'][0]))

    summary_ids_lst = [model.generate(inputs, num_beams=4, max_length=1024, early_stopping=True) for inputs in inputs_batch_lst]

    summary_batch_lst = []
    for summary_id in summary_ids_lst:
        summary_batch = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
        summary_batch_lst.append(summary_batch[0])

    summary_all = '\n'.join(summary_batch_lst)
    return summary_all


def create_pdf_report(headers, text, tokenizer, model, size, images_folder='images', output_pdf='summary.pdf'):
    """
    Create a PDF report with headers, summarized text, and images.

    This function processes the provided headers and text, generates summaries using the provided
    tokenizer and model, and includes images from the specified folder. The final report is saved
    as a PDF file.

    Args:
        headers (list): A list of headers found in the text.
        text (str): The main body of text to summarize and include in the report.
        tokenizer: The tokenizer to preprocess the input text.
        model: The pre-trained model to generate the summaries.
        size (int): The scaling factor for images.
        images_folder (str, optional): The folder containing images to include in the report. Defaults to 'images'.
        output_pdf (str, optional): The file path for the output PDF. Defaults to 'summary.pdf'.
    """
    patternForBrackets = re.compile(r'\[\s*\d+(?:,\s*\d+)*\s*]')
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    styles = getSampleStyleSheet()
    style = styles['Normal']
    elements = []

    for header in range(len(headers) - 16):
        startHeader = headers[header]
        endHeader = headers[header + 1]

        dot_count = startHeader.count('.')
        if dot_count == 0:
            header_style = styles['Heading2']
        elif dot_count != 0:
            header_style = styles['Heading5']

        pattern = re.compile(re.escape(startHeader) + "(.*?)" + re.escape(endHeader), re.DOTALL)
        match = pattern.search(text)

        if match:
            text_between = match.group(1)
            text_without_brackets = patternForBrackets.sub('', text_between)
            text_final = text_without_brackets.replace("\\n", "").replace("\n", "")

            try:
                summary_text = summarize_text(text_final, tokenizer, model)
            except ValueError as e:
                print(f"Error: {e}")
            except IndexError as e:
                print(f"Index Error: {e}")

            elements.append(Paragraph(startHeader, header_style))
            elements.append(Paragraph(summary_text, style))
            elements.append(Spacer(1, 12))

    photo_index = len(os.listdir(images_folder))
    if photo_index > 0:
        for img_num in range(photo_index):
            image_path = os.path.join(images_folder, f'image_{img_num}.png')
            if os.path.exists(image_path):
                pil_image = Image.open(image_path)
                real_width, real_height = pil_image.size

                dpi = 72
                size = size
                width_in_points = real_width / dpi * size
                height_in_points = real_height / dpi * size

                img = ReportLabImage(image_path, width=width_in_points, height=height_in_points)
                elements.append(img)

                centered_style = ParagraphStyle(name='CenteredStyle', parent=styles['Normal'], alignment=TA_CENTER)
                elements.append(Paragraph(f"Figure {img_num + 1}", centered_style))
                elements.append(Spacer(1, 12))

    doc.build(elements)


def main(link, is_photo=None, size=None):
    pdf_path = download_pdf(link)
    text = extract_text_from_pdf(pdf_path)
    print("Text extracted successfully")

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
    pdf_reader = LayoutPDFReader(llmsherpa_api_url)
    doc = pdf_reader.read_pdf(pdf_path)
    doc = doc.to_html()
    print("HTML document created successfully")

    headers = extract_headers_from_html(doc)
    print("Headers extracted successfully")

    if is_photo:
        extract_images_from_pdf(pdf_path)

    print("Creating PDF summary")
    create_pdf_report(headers, text, tokenizer, model, size)
    print("PDF summary created successfully")
  # Save data to MongoDB
    mongo_saver = MongoDBSaver("mongodb://localhost:27017/", "arxiv_db", "documents")
    summary_data = {
        "link": link,
        "headers": headers,
        "text": text,
        "summary": summarize_text(text, tokenizer, model)
    }
    mongo_saver.save_data(summary_data)
    print("Data saved to MongoDB successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a string and an optional object.")

    parser.add_argument('link', type=str, help='A required input link to arXiv doc')
    parser.add_argument('--p', type=bool, help='Optional flag for image extracting', default=True)
    parser.add_argument('--s', type=int, help='Optional flag for image size', default=7)

    args = parser.parse_args()

    main(args.link, args.p, args.s)
