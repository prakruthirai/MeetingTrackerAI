# Install the transformers library
!pip install transformers

# Install the python-docx library
!pip install python-docx

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("summarization", model="kabita-choudhary/finetuned-bart-for-conversation-summary")

# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from docx import Document

# Load pre-trained BART model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")
model = AutoModelForSeq2SeqLM.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")

# Get a list of all .txt files in the directory
txt_files = [file for file in os.listdir() if file.endswith(".txt")]

# If there are no .txt files, exit the script
if not txt_files:
    print("No .txt files found in the directory.")
    exit()

# Select the most recently modified .txt file
most_recent_file = max(txt_files, key=os.path.getctime)

# Read conversation from the most recently modified text file
with open(most_recent_file, "r") as file:
    input_text = file.read()

# Tokenize and prepare input for BART
input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=1024, truncation=True)

# Generate summary
summary_ids = model.generate(input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Print the generated summary
print("\nGenerated Summary:\n", summary)

# Save the summary as a .docx file
output_docx_file = "summary.docx"
doc = Document()
doc.add_paragraph(summary)
doc.save(output_docx_file)

print(f"Summary saved as {output_docx_file}")
