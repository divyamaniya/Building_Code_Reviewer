'''
This file is reponsible for taking raw data provided by the loader and output structured objects
This file also has a defination of all specialized loader; for now only supporting pdf documents

'''

import os
from src.ingestion.base import BaseParser
import logging

# it can be docling, unstructured, llamaparse... any
class DoclingParser(BaseParser):        # it is wrapper around docling, why do we implement it?
    def __init__(self):             # 26 times object created
        logging.info("DoclingParser object created")

    def load_docling_advanced_option(self):             # 26 times this function called
        logging.info("DoclingParser option load function")
        # hardware selection
        from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice
        accel_options = AcceleratorOptions(
            device = AcceleratorDevice.CUDA,
            num_threads = os.cpu_count()
        )

        # Pipeline options
        from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
        pipeline_options = ThreadedPdfPipelineOptions(
            accelerator_options = accel_options,
            page_batch_size = 16,
            do_ocr= True,
            do_table_structure= True
        )

        # Assemble options in DocumentConverter
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend)
            }
        )

        return converter
        

    def parse(self, file_path: str):
        # need to change all logging to f .. {} or %s, var
        logging.debug("DoclingParser parse function with file: %s", file_path)
        converter = self.load_docling_advanced_option()
        result = converter.convert(file_path)
        return result

class UnstructuredParser(BaseParser):
    
    def parser(self, file_path: str):
        pass