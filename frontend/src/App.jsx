import React, { useState } from 'react';
import ChatInterface from './components/ChatInterface';
import PDFViewer from './components/PDFViewer';

export default function App() {
    const [activeCitation, setActiveCitation] = useState(null);

    return (
        <div className="flex h-screen w-screen overflow-hidden bg-gray-950">
            {/* Left side: AI Chat Interface */}
            <div className="w-1/2 h-full flex flex-col p-4">
                <header className="mb-4">
                    <h1 className="text-xl font-bold text-white">RAG Document Assistant</h1>
                    <p className="text-xs text-gray-400">Connected to local splitted PDF storage</p>
                </header>
                <div className="flex-1 overflow-hidden">
                    <ChatInterface onSelectCitation={(citation) => setActiveCitation(citation)} />
                </div>
            </div>

            {/* Right side: Side-by-Side PDF Viewer & Highlight Fallback */}
            <div className="w-1/2 h-full flex flex-col p-4">
                <div className="flex-1 overflow-hidden">
                    <PDFViewer activeCitation={activeCitation} />
                </div>
            </div>
        </div>
    );
}