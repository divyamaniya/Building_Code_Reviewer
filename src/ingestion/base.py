'''
This file provide template to all specialize class that ingest data from different kind of sources, viz pdf, md, image, video
'''


from abc import ABC, abstractmethod


class BaseLoader(ABC):
    @abstractmethod
    def load(self, source:str):
        pass

class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str):
        pass

class BaseChunker(ABC):
    import re
    from docling_core.types.doc import DoclingDocument

    @abstractmethod
    def chunk(self, document_obj: DoclingDocument) -> list:
        pass

    def _get_page_offset(self, filename: str) -> int:
        '''Extracts 50 from file_name_offset_50.json'''
        match = self.re.search(r"offset_(\d+)", filename)
        return int(match.group(1)) if match else 0
    
    def _apply_virtual_offset(self, data: dict, offset: int):
        '''
        when we split pdf, each pdf starts with page no 1, thus docling object also have page_number 1 however actual page could be 51 or 201
        so this function Reset Page Numbers inside DoclingDocument object metadata
        '''
        if offset == 0:
            return data
        
        def adjust(obj):                    # recursive helper; because Docling object is a "tree" of nested lists and dictonaries, so we need function that can check every corner of the tree.
            # implicit base condition if not list or dict it simply returns
            if isinstance(obj, list):       # if current part is list of paragraph, it loops through every item
                for item in obj: adjust(item)
            elif isinstance(obj, dict):     # if current part is dict, then check page_no and update count
                if "page_no" in obj:
                    obj["page_no"] += offset
                for val in obj.values():
                    adjust(val)

        adjust(data)
        # one page can have hundreds of bounding boxes, text fragments, and metadata tages.
        # recursion in python have to travers every nested leve. for 50 page chunk, you might trigger thousands of function calls.

        # improved iternative version:
        # stack = [data]
        # while stack:
        #     current = stack.pop()
        #     if isinstance(current, dict):
        #         if "page_no" in current:
        #             current["page_no"] += offset
        #         stack.extend(current.values())
        #     elif isinstance(current, list):
        #         stack.extend(current)

        return data
        
    def run_on_file(self, json_path: str):
        '''This file is a wrapper to load, offset and chunk any Docling JSON'''
        import json

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # detect and apply virtual offset
        # offset = self._get_page_offset(json_path)
        # if offset > 0:
            # data = self._apply_virtual_offset(data, offset)

        # converting json back to docling object
        doc_obj = self.DoclingDocument.model_validate(data)
        
        return self.chunk(doc_obj)
    

class BaseVectorStore(ABC):
    @abstractmethod
    # def upsert_chunks(self, collection_name, chunks, embeddings):
    #     '''method to insert or update embedding of chunks'''
    #     pass
    def add_documents(self, chunks):
        pass

    
    @abstractmethod
    def delete_collection(self):
        pass