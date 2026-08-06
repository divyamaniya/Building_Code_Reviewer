'''This file' primary job is I/O, bringing all required files into memory'''

from pathlib import Path
import pypdfium2 as pdfium
from src.ingestion.base import BaseLoader

import logging
logger = logging.getLogger(__name__)

class LocalPDFLoader(BaseLoader):
    def __init__(self):
        # self.source_folder = Path(source_folder)
        pass

    def get_all_pdfs(self, source_folder):
        '''
        This function will check all the pdf available give source folder and sub folder
        '''
        pdf_files = list(Path(source_folder).rglob("*.pdf"))
        logger.debug("List of pdf files: %s", pdf_files)
        if not pdf_files:
            print(f"No PDFs found in any of the folder in {self.source_folder}")
        return pdf_files
    

    def load(self, source: str):
        # logic should return all the list of documents
        # more advanced it will check for only new files to embedd, and only modified files

        raw_paths = self.get_all_pdfs(source)
        pdf_file_paths = []

        # splitting file if very large fiel
        for path in raw_paths:
            doc = pdfium.PdfDocument(path)
            page_count = len(doc)
            doc.close()

            if page_count > 500:
                logger.info(f"Large file detected ({page_count} pages). Splitting virtually...")
                segments = VirtualPDFSplitter(Path(source).parent/'temp_data/splitted').split(path)
                pdf_file_paths.extend(segments)
            else:
                pdf_file_paths.append(path)
        
        return pdf_file_paths

        



class VirtualPDFSplitter:           # if we have very big pdfs (1000 page), to speed up process we split it into multiple parts
    def __init__(self, output_dir: str, pages_per_segment: int=50):
        self.output_dir = Path(output_dir)
        self.pages_per_segment = pages_per_segment
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def split(self, pdf_path: str):
        logger.info(f"pdf path is : {pdf_path}")
        src_pdf = pdfium.PdfDocument(pdf_path)
        total_pages = len(src_pdf)
        base_name = Path(pdf_path).stem

        segments = []
        for start_page in range(0, total_pages, self.pages_per_segment):
            end_page = min(start_page + self.pages_per_segment, total_pages)
            dest_pdf = pdfium.PdfDocument.new()

            page_indices = [i for i in range(start_page, end_page)]
            dest_pdf.import_pages(src_pdf, page_indices)

            output_filename = f"{base_name}_offset_{start_page}.pdf"
            output_path = self.output_dir / output_filename

            dest_pdf.save(output_path)
            segments.append(str(output_path))

            dest_pdf.close()
        src_pdf.close()
        return segments



class WebLoader(BaseLoader):
    def load(self, source: str):
        # logic should return all the list of HTML/Internet pages
        pass


class GoogleDriveLoader(BaseLoader):
    # This will establish connection with google drive and read all the files
    pass