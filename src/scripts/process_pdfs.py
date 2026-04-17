import os
import time
from pathlib import Path
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat

def convert_pdfs_to_md(source_folder, output_folder):
    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    accel_options = AcceleratorOptions(
        device=AcceleratorDevice.CUDA,
        num_threads=os.cpu_count()
    )
    print("Accel Options: ", accel_options)

    pipeline_options = ThreadedPdfPipelineOptions(
        accelerator_options = accel_options,
        page_batch_size = 16,
        do_ocr=True,
        do_table_structure=True,
        images_scale=1.0
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend)
        }
    )

    pdf_files = list(Path(source_folder).rglob("*.pdf"))
    print(f"Starting conversion of {len(pdf_files)} files to '{output_folder}'...")

    for result in converter.convert_all(pdf_files):
        if result.document:
            file_name = f"{Path(result.input.file).stem}.md"
            final_output_path = out_path / file_name

            markdown_text = result.document.export_to_markdown()
            with open(final_output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)

            print(f"Saved: {final_output_path}")
        else:
            print(f"Error processing: {result.input.file}")


if __name__ == "__main__":
    raw_data_path = "./data/raw_data/"
    output_data_path = "./data/processed_data/speed_up_try/"
    start_time = time.perf_counter()
    convert_pdfs_to_md(raw_data_path, output_data_path)
    end_time = time.perf_counter()
    total_duration = end_time - start_time

    print(f"Total processing time: {total_duration:.2f} seconds")
    print(f"Average time per document: {total_duration / 2:.2f} seconds")
    
    # page_batch_size: process 100 batch page
    # Process bar: 
    # from tqdm import tqdm
    # for result in tqdm(converter.convert_all(pdf_files), total=len(pdf_files), desc="Converting PDFs"):