from typing import List, Dict, Any
from src.ingestion.base import BaseChunker
from src.utils.loader import LocalModelLoader

import logging

logger = logging.getLogger(__name__)
# in the split pdf case, we also need to check the last page of the last split document

class CitationMixin:

    @staticmethod
    def _bbox_to_coords(bbox: dict) -> dict:
        '''Normalizes a Docling BoundingBox dict {l, t, r, b, ...} into the
        HighlightCoords-compatible shape {x1, y1, x2, y2} used by schema.py.'''
        return {
            "x1": float(bbox.get("l", 0.0)),
            "y1": float(bbox.get("t", 0.0)),
            "x2": float(bbox.get("r", 0.0)),
            "y2": float(bbox.get("b", 0.0)),
        }

    @classmethod
    def _extract_citation_metadata(cls, doc_items: list) -> dict:
        '''
        returns:
            {"page_number": int, "highlight_coords": {"x1","y1","x2","y2"} | None}
        '''
        page_number = 0
        merged_bbox = None

        for item in doc_items or []:
            for prov in item.get("prov", []) or []:
                page_no = prov.get("page_no")
                if page_no is not None and page_number == 0:
                    page_number = int(page_no)

                bbox = prov.get("bbox")
                if not bbox:
                    continue

                coords = cls._bbox_to_coords(bbox)
                if merged_bbox is None:
                    merged_bbox = coords
                else:
                    merged_bbox["x1"] = min(merged_bbox["x1"], coords["x1"])
                    merged_bbox["y1"] = min(merged_bbox["y1"], coords["y1"])
                    merged_bbox["x2"] = max(merged_bbox["x2"], coords["x2"])
                    merged_bbox["y2"] = max(merged_bbox["y2"], coords["y2"])

        return {"page_number": page_number, "highlight_coords": merged_bbox}

    @staticmethod
    def _item_to_prov_dict(item) -> dict:
        '''Converts Docling item object into the plain-dict `doc_item`
        shape `_extract_citation_metadata` expects, so both code paths
        (attached doc_items metadata, and this offset-map alignment) can
        share one merge implementation.'''
        prov_list = []
        for prov in getattr(item, "prov", []) or []:
            bbox = getattr(prov, "bbox", None)
            prov_list.append({
                "page_no": getattr(prov, "page_no", None),
                "bbox": {
                    "l": bbox.l, "t": bbox.t, "r": bbox.r, "b": bbox.b
                } if bbox is not None else None
            })
        return {"prov": prov_list}

    @classmethod
    def _build_offset_item_map(cls, doc_obj, md_text: str) -> list:

        offset_map = []
        cursor = 0

        try:
            iterator = doc_obj.iterate_items()
        except Exception as e:
            logger.warning(f"Could not iterate Docling items for offset alignment: {e}")
            return offset_map

        for item, _level in iterator:
            item_text = getattr(item, "text", None)
            if not item_text:
                continue  # tables/pictures: no plain text to align on

            idx = md_text.find(item_text, cursor)
            if idx == -1:
                # Formatting (headers, bullets, escaping) occasionally breaks
                # an in-order match; retry once from the top before giving up
                # on this item rather than desyncing every item after it.
                idx = md_text.find(item_text)
                if idx == -1:
                    continue

            start, end = idx, idx + len(item_text)
            cursor = end
            offset_map.append((start, end, cls._item_to_prov_dict(item)))

        return offset_map

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
    

class MarkdownChainedChunker(BaseChunker, CitationMixin):
    markdown_params = {
        "image_placeholder": "[[IMAGE_BLOCK]]",
        "page_break_placeholder": "[[PAGE_BREAK]]",
        # "traverse_pictures": True,
        # "compact_tables": True,
        # "escape_html": True
    }
    current_page_markers = [] # State storage for page mapping
    current_item_offsets = [] # State storage for markdown-offset -> Docling item alignment

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

        # 2. Also build the char-offset -> Docling item alignment, so that
        # text-splitter chunkers (which only ever see this flat markdown
        # string, not the source doc_items) can still recover precise
        # highlight_coords for each chunk in _format_markdown_output.
        self.current_item_offsets = self._build_offset_item_map(doc_obj, md_text)

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
            end_char = n.end_char_idx if n.end_char_idx is not None else start_char
            
            # Find the last marker that appeared BEFORE this chunk's start
            current_page = 0
            for marker_pos, page_num in self.current_page_markers:
                if marker_pos <= start_char:
                    current_page = page_num
                else:
                    break

            # Citation metadata, in order of preference:
            #   1. `doc_items` still attached to the node's own metadata
            #      (StructureAwareChunker, via DoclingNodeParser) - most direct.
            #   2. Our own char-offset -> item alignment (Recursive/Semantic/
            #      SlidingWindow), intersecting this chunk's [start_char, end_char)
            #      against `current_item_offsets` built in _get_markdown_doc.
            #   3. Fall back to the page-level heuristic above, no bbox.
            citation_meta = {"page_number": current_page, "highlight_coords": None}

            doc_items = n.metadata.get("doc_items")
            if doc_items:
                extracted = self._extract_citation_metadata(doc_items)
            else:
                overlapping_items = [
                    item for (item_start, item_end, item) in self.current_item_offsets
                    if item_start < end_char and item_end > start_char
                ]
                extracted = self._extract_citation_metadata(overlapping_items) if overlapping_items else None

            if extracted:
                if extracted["page_number"]:
                    citation_meta["page_number"] = extracted["page_number"]
                if extracted["highlight_coords"]:
                    citation_meta["highlight_coords"] = extracted["highlight_coords"]

            processed.append({
                "text": text.replace("[[PAGE_BREAK]]", ""), 
                "metadata": {
                    **n.metadata,
                    **citation_meta,
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
                        
            # 1. Determine Page using markers stored during _get_markdown_doc
            start_char = getattr(n, 'start_char_idx', None)
            if start_char is None and hasattr(n, 'text_resource'): # fallback safety
                start_char = 0
            start_char = start_char if start_char is not None else 0
            
            end_char = getattr(n, 'end_char_idx', None)
            end_char = end_char if end_char is not None else start_char

            current_page = 0
            for marker_pos, page_num in self.current_page_markers:
                if marker_pos <= start_char:
                    current_page = page_num
                else:
                    break

            citation_meta = {"page_number": current_page, "highlight_coords": None}

            # 2. Extract citation metadata via doc_items or char-offset fallback mapping
            doc_items = n.metadata.get("doc_items")
            if doc_items:
                extracted = self._extract_citation_metadata(doc_items)
            else:
                overlapping_items = [
                    item for (item_start, item_end, item) in self.current_item_offsets
                    if item_start < end_char and item_end > start_char
                ]
                extracted = self._extract_citation_metadata(overlapping_items) if overlapping_items else None

            if extracted:
                if extracted["page_number"]:
                    citation_meta["page_number"] = extracted["page_number"]
                if extracted["highlight_coords"]:
                    citation_meta["highlight_coords"] = extracted["highlight_coords"]

            processed.append({
                "text": n.get_content().replace("[[PAGE_BREAK]]", ""),
                "node_id": n.node_id,
                "metadata": {
                    **n.metadata,
                    **citation_meta,
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
            doc_items = meta.get("doc_items", [])

            # Precise citation metadata straight from Docling's own chunk
            # provenance. Unlike the previous implementation (which only read
            # doc_items[0]), this merges provenance across every doc_item that
            # contributed to the chunk, so multi-item chunks (e.g. a heading +
            # paragraph, or several table rows) still get an accurate bounding
            # envelope instead of just the first item's box.
            citation_meta = self._extract_citation_metadata(doc_items)

            processed.append({
                "text": text,
                "metadata": {
                    # **meta,
                    "page_number": citation_meta["page_number"],
                    "highlight_coords": citation_meta["highlight_coords"],
                    "is_hybrid": True,
                    "dl_path": doc_items[0].get("self_ref", "unknown") if doc_items else "unknown"
                }
            })
        return processed