from PyPDF2 import PdfReader, PdfWriter

def split_pdf_in_two(input_pdf_path, output_pdf1_path, output_pdf2_path):
    reader = PdfReader(input_pdf_path)
    total_pages = len(reader.pages)
    midpoint = total_pages // 2

    writer1 = PdfWriter()
    writer2 = PdfWriter()

    # First half
    for i in range(midpoint):
        writer1.add_page(reader.pages[i])
    with open(output_pdf1_path, 'wb') as f:
        writer1.write(f)

    # Second half
    for i in range(midpoint, total_pages):
        writer2.add_page(reader.pages[i])
    with open(output_pdf2_path, 'wb') as f:
        writer2.write(f)

    print(f"PDF split into two parts: {output_pdf1_path}, {output_pdf2_path}")

# Example usage
split_pdf_in_two("input.pdf", "part1.pdf", "part2.pdf")
