#!/bin/bash

# Function to trim PDF borders using pdfcrop
trim_pdf_borders() {
    input_file=$1
    output_file="${input_file%.pdf}-cropped.pdf"
    pdfcrop "$input_file" "$output_file"
}

# Directory containing the PDF files
input_directory="../plots"

# Process each PDF file in the directory
for pdf_file in "$input_directory"/*.pdf; do
    trim_pdf_borders "$pdf_file"
done

# Move trimmed pdfs in the output directory
output_directory="../plots/trimmed"
mkdir -p "$output_directory"
mv "$input_directory"/*-cropped.pdf "$output_directory"

echo "All PDFs have been processed and borders trimmed."
