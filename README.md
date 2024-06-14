# Overview
PDF-to-PDF arXiv document summarisation program with images and headers extraction. Example, the well-known article "Attention is all you need", document - [arXiv_doc](https://github.com/KyryloTurchyn/arXiv_summarizer/blob/main/arXiv_doc.pdf) and its summarisation [summary](https://github.com/KyryloTurchyn/arXiv_summarizer/blob/main/summary.pdf).

# Installation 
```
git clone https://github.com/KyryloTurchyn/arXiv_summarizer
pip install -r requirements.txt
```
To run script:
```
python main.py <link> [--p <bool>] [--s <int>]
```
where 
  - `link` - link to arXiv document (e.g. "https://arxiv.org/pdf/1706.03762.pdf")
  - `--p` - optional flag for image extraction
  - `--s` - optional flag for size of image 
# Description
The essence of the program is to search for headings, summarise the text from heading to heading and finally create a PDF document with summarised text and all images from the original article. In this work we used the Facebook BART model for summarisation, as well as the BART tokeniser.

# About program
This paragraph will describe the structure of the programme point by point, explaining the main idea and the process of this point.
> Also there will be comments-explanations in the code itself
  ## Headers and photo extraction
The main idea in extracting all the headers we need is to convert the original PDF file to HTML format. In HTML format, all headings will be labelled with standard header tags like < h1 >, < h2 > etc and the < p > tag. Using parsing technology and BeautifulSoup, we get all the strings that have the tags we need. 

After that, we need to select from all the collected headers, the ones we need. To do this, we can apply regex and take out all the headers we need.

We get the photos by requesting them on the site and just saving them in a folder.

  ## Text summarization
    
  This Python function, named summarize, leverages a tokenizer and a neural model to generate summaries for a given input text. The function is structured to handle large inputs that exceed the model's maximum token limit by breaking the text into manageable chunks without truncation at word boundaries.
  
  How It Works:
  #### Tokenization without Truncation:
  The input text is tokenized using a specified tokenizer with no maximum length limit (`max_length=None`) and without truncation (`truncation=False`). This ensures that all the input text is tokenized for processing. 
  #### Chunking:
  The tokenized text is divided into chunks. Each chunk size is determined by the model's maximum token length capability (e.g., 1024 tokens for BART). The chunking process respects word boundaries by ensuring that chunks end at the last complete word within the limit.  
  #### Batch Preparation:
  Each chunk is wrapped in a PyTorch tensor and added to a batch list (`inputs_batch_lst`). This list of tensors is prepared for batch processing by the model.  
  #### Summary Generation:
  The model generates a summary for each tensor in the batch list using a beam search strategy with four beams (`num_beams=4)`. The `max_length` parameter limits the summary length, and `early_stopping` improves efficiency by halting the generation once a satisfactory summary end is reached.  
  #### Decoding and Aggregation:
  Each generated summary, represented by token IDs, is decoded back into text, skipping any special tokens and maintaining original tokenization spaces. The decoded summaries are collected into a list (`summary_batch_lst`).  
  #### Concatenation:
  All individual summaries are concatenated into a single string with newline characters separating each summary, forming the final comprehensive summary output.  
  #### Return Value:
  The function returns the concatenated summary of the entire input text, making it suitable for use cases where a quick and coherent overview of a large body of text is required.

  ## Text and photo adding
  Create an array elements array to store all data that will be written to the PDF document in the future. First we summarise the text from header to header and add the header itself to elements and then the text.
  After that we optionally add images
# Notes
#### 1) The programme does not work on documents where the text format is given as columnar format
#### 2) There may be a problem with images, as the image size may be too large. For this reason I have added `--s` flag.
