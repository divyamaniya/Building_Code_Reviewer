import React, { useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';

pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

export default function PDFViewer({ activeCitation }) {
    const [numPages, setNumPages] = useState(null);
    const [pageNumber, setPageNumber] = useState(1);

    // Sync active citation page if selected
    const currentDoc = activeCitation?.source_doc ? `/api/documents/${activeCitation.source_doc}` : null;
    const targetPage = activeCitation?.page_number || pageNumber;
    const hasPrecise = activeCitation?.has_precise_highlight || false;

    return (
        <div className="flex flex-col h-full bg-gray-950 border-l border-gray-800 p-4">
            <div className="flex justify-between items-center mb-4 bg-gray-900 p-3 rounded-lg border border-gray-800">
                <span className="text-sm font-medium text-gray-300 truncate">
                    {activeCitation ? activeCitation.source_doc : 'No document selected'}
                </span>
                {!hasPrecise && activeCitation && (
                    <span className="text-xs bg-yellow-900 text-yellow-200 px-2 py-1 rounded">
                        Fallback: Page-level view (Precise bounding unavailable)
                    </span>
                )}
            </div>

            <div className="flex-1 overflow-auto flex justify-center bg-gray-900 rounded-lg border border-gray-800 p-2 relative">
                {currentDoc ? (
                    <Document
                        file={currentDoc}
                        onLoadSuccess={({ numPages }) => setNumPages(numPages)}
                        loading={<div className="text-gray-400">Loading document canvas...</div>}
                    >
                        <Page pageNumber={targetPage} renderTextLayer={true} renderAnnotationLayer={true} />
                        
                        {/* Precise Highlight Overlay Rendering */}
                        {hasPrecise && activeCitation.coord_x1 != null && (
                            <div
                                style={{
                                    position: 'absolute',
                                    left: `${activeCitation.coord_x1 * 100}%`,
                                    top: `${activeCitation.coord_y1 * 100}%`,
                                    width: `${(activeCitation.coord_x2 - activeCitation.coord_x1) * 100}%`,
                                    height: `${(activeCitation.coord_y2 - activeCitation.coord_y1) * 100}%`,
                                    backgroundColor: 'rgba(255, 235, 59, 0.4)',
                                    border: '2px solid rgba(255, 193, 7, 0.8)',
                                    pointerEvents: 'none',
                                    transition: 'all 0.3s ease-in-out'
                                }}
                            />
                        )}
                    </Document>
                ) : (
                    <div className="flex items-center justify-center text-gray-500 h-full">
                        Select a citation source to load document preview.
                    </div>
                )}
            </div>
        </div>
    );
}