from typing import List, Dict, Any
from src.ingestion.base import BaseChunker
from src.utils.loader import LocalModelLoader

import logging

logger = logging.getLogger(__name__)
# in the split pdf case, we also need to check the last page of the last split document

# export_to_markdown() it converts the rich, structured DoclingDocument into a flat string.
# so we are going to convert into DoclingNodeParser first
class LlamaChainedChunker(BaseChunker):
    '''This is Base class for LlamaIndex techniques that requires Page-level metadata preservation.'''
    
    def _get_atomic_nodes(self, doc_obj):       # one node per paragraph, table or header
        # llama_index expect the document object to have "id_" so this function transform docling object into required Llama Document object
        from llama_index.node_parser.docling import DoclingNodeParser
        import json
        from llama_index.core.schema import Document as LIDocument

        doc_json_str = json.dumps(doc_obj.export_to_dict())
        
        # 2. Wrap it in a LlamaIndex Document object
        # We give it an ID and pass the JSON as the main content
        li_doc = LIDocument(
            text=doc_json_str,
            id_=getattr(doc_obj.origin, 'filename', 'source_doc') # Provides the missing id_
        )
        logger.debug(f"Llama index document: {li_doc}")
        parser = DoclingNodeParser()
        logger.debug(f"After DoclingNodeParser: {parser.get_nodes_from_documents([li_doc])}")
        return parser.get_nodes_from_documents([li_doc])

    def _format_output(self, nodes) -> List[Dict[str, Any]]:
        return [
            {
                "text": n.get_content(),
                "metadata": {
                    **n.metadata,
                    "node_id": n.node_id
                }
            }for n in nodes
        ]
    

class MarkdownChainedChunker(BaseChunker):
    markdown_params = {
        "image_placeholder": "[[IMAGE_BLOCK]]",
        "page_break_placeholder": "[[PAGE_BREAK]]",
        # "traverse_pictures": True,
        # "compact_tables": True,
        # "escape_html": True
    }
    current_page_markers = [] # State storage for page mapping

    def _get_markdown_doc(self, doc_obj):
        import re
        from llama_index.core.schema import Document as LIDocument
        md_text = doc_obj.export_to_markdown(**self.markdown_params)

        # 1. Map character positions to page numbers BEFORE chunking
        # We store this in the class instance so _format_markdown_output can see it
        self.current_page_markers = [
            (m.start(), i + 1) 
            for i, m in enumerate(re.finditer(r"\[\[PAGE_BREAK\]\]", md_text))
        ]

        # logger.debug(f"Exported Markdown: {md_text}")
        return LIDocument(
            text = md_text,
            meta_data={"source_doc": getattr(doc_obj.origin, 'filename', 'unknown')}
        )
    
    def _format_markdown_output(self, nodes):
        import re
        processed = []
        for n in nodes:
            text = n.get_content()
            
            # 2. Determine Page: Use the markers stored during _get_markdown_doc
            start_char = n.start_char_idx if n.start_char_idx is not None else 0
            
            # Find the last marker that appeared BEFORE this chunk's start
            current_page = 0
            for marker_pos, page_num in self.current_page_markers:
                if marker_pos <= start_char:
                    current_page = page_num
                else:
                    break

            processed.append({
                "text": text.replace("[[PAGE_BREAK]]", ""), 
                "metadata": {
                    **n.metadata,
                    "page_number": current_page,
                    "contains_image": "[[IMAGE_BLOCK]]" in text,
                    "is_markdown": True,
                    "node_id": n.node_id
                }
            })
        return processed
    
    def _format_hierarchical_ouput(self, nodes):
        processed = []
        for n in nodes:
            rels = {}       # contains extract relationships (Parent, Children, Next, Previous)
            if hasattr(n, 'relationships'):
                for key, val in n.relationships.items():
                    if val is None:
                        continue

                    rel_type = str(key) # 1: Source, 2: Next, 3: Previous, 4: Parent, 5: Children
                    # FIX: Handle cases where val is a list (typical for children)
                    if isinstance(val, list):
                        # Store list of IDs if there are multiple children
                        rels[rel_type] = [item.node_id for item in val if hasattr(item, 'node_id')]
                    else:
                        # Store single ID for Parent/Prev/Next
                        rels[rel_type] = getattr(val, 'node_id', None)
            
            processed.append({
                "text": n.get_content().replace("[[PAGE_BREAK]]", ""),
                "node_id": n.node_id,
                "metadata": {
                    **n.metadata,
                    "page_number": self._calculate_page(getattr(n, 'start_char_idx', 0)) if hasattr(self, '_calculate_page') else 0,
                    "is_hierarchical": True
                },
                "relationships": rels
            })
        return processed
        



class RecursiveChunker(MarkdownChainedChunker):
    """Standard intelligent splitting that respects sentence boundaries."""
    def __init__(self, chunk_size: int=500, chunk_overlap: int = 50):
        # super().__init__()
        from llama_index.core.node_parser import SentenceSplitter
        self.chunker = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk(self, doc_obj):
        # nodes = self.chunker.get_nodes_from_documents(self._get_atomic_nodes(doc_obj))
        # return self._format_output(nodes)
        li_doc = self._get_markdown_doc(doc_obj)
        nodes = self.chunker.get_nodes_from_documents([li_doc])
        return self._format_markdown_output(nodes)

class SemanticChunker(MarkdownChainedChunker):
    """Splits based on meaning shifts using embeddings. High accuracy, slower speed."""
    def __init__(self, embed_model):
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        # Need to implement llamaindex compatible embedding object
        self.embed_model = LocalModelLoader().load_embedding(embed_model)
        self.chunker = SemanticSplitterNodeParser(buffer_size=5, embed_model=self.embed_model)

    def chunk(self, doc_obj):
        # nodes = self.chunker.get_nodes_from_documents(self._get_atomic_nodes(doc_obj))
        # return self._format_output(nodes)
        li_doc = self._get_markdown_doc(doc_obj)
        nodes = self.chunker.get_nodes_from_documents([li_doc])
        return self._format_markdown_output(nodes)
        

class SlidingWindowChunker(MarkdownChainedChunker):
    """Strict token-count based splitting with overlap."""
    def __init__(self, chunk_size: int=512, chunk_overlap: int= 128):
        # super().__init__()
        from llama_index.core.node_parser import TokenTextSplitter
        self.chunker = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk(self, doc_obj):
        # nodes = self.chunker.get_nodes_from_documents(self._get_atomic_nodes(doc_obj))
        # return self._format_output(nodes)
        li_doc = self._get_markdown_doc(doc_obj)
        nodes = self.chunker.get_nodes_from_documents([li_doc])
        return self._format_markdown_output(nodes)
        

class HierarchicalChunker(MarkdownChainedChunker):
    """Creates Parent-Child relationships for summary + detail retrieval."""
    def __init__(self, chunk_sizes=[2048, 512, 128]):
        # it produces multiple 'levels' of the same text.
        # once in large parent chunk and again in sevral small child chunks
        # can also be seen as parent-child relationship; where parents(large chunk) and children (small chunk)
        from llama_index.core.node_parser import HierarchicalNodeParser
        self.chunker = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)

    def chunk(self, doc_obj):
        # Note: Hierarchical returns more complex nodes; we simplify for JSON output
        # nodes = self.chunker.get_nodes_from_documents(self._get_atomic_nodes(doc_obj))
        # return self._format_output(nodes)
        li_doc = self._get_markdown_doc(doc_obj)
        nodes = self.chunker.get_nodes_from_documents([li_doc])
        return self._format_hierarchical_ouput(nodes)

class StructureAwareChunker(MarkdownChainedChunker):
    """Directly uses Docling's hierarchy but through LlamaIndex's parser."""
    """Ensures that tables, lists and headers are kept as atomic units."""
    # One Node = One Paragraph/Table
    def __init__(self):
        from llama_index.node_parser.docling import DoclingNodeParser   # DoclingNodeParser respects document structure (paragraphs, headings, tables)
        self.chunker = DoclingNodeParser()
    
    def chunk(self, doc_obj):
        import json
        from llama_index.core.schema import Document as LIDocument
        # The DoclingNodeParser is unique because it can take the docling object directly
        # return self._format_output(self._get_atomic_nodes(doc_obj))
        doc_json_str = json.dumps(doc_obj.export_to_dict())
        li_doc = LIDocument(
            text = doc_json_str,
            id_ = getattr(doc_obj.origin, 'filename', 'source_doc')
        )
        nodes = self.chunker.get_nodes_from_documents([li_doc])
        return self._format_markdown_output(nodes)

class DoclingHybridChunker(MarkdownChainedChunker):
    """Docling Native: Does not use LlamaIndex parser."""
    """Contextualization by merging headings with text"""
    # One Node = Text + Its Headings

    def __init__(self, tokenizer_name:str = "sentence-transformers/all-MiniLM-L6-v2"):
        from docling.chunking import HybridChunker
        self.strategy = HybridChunker(tokenizer=tokenizer_name)     # we can pass tokenizer to ensure chunks fit embedding model's limits

    def chunk(self, doc_obj):
        chunks = self.strategy.chunk(doc_obj)
        
        processed = []
        for c in chunks:
            text = self.strategy.contextualize(c)
            meta = c.meta.export_json_dict()
            processed.append({
                "text": text,
                "metadata": {
                    # **meta,
                    "page_number": meta.get("doc_items", [{}])[0].get("prov", [{}])[0].get("page_no", 0),
                    "is_hybrid": True,
                    "dl_path": meta.get("doc_items", [{}])[0].get("self_ref", "unknown")
                }
            })
        return processed