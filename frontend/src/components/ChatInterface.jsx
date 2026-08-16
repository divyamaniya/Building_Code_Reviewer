import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

export default function ChatInterface({ onSelectCitation }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (!input.trim() || loading) return;

        const userMsg = { role: 'user', content: input };
        setMessages((prev) => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            const res = await axios.post('http://localhost:8000/api/chat', {
                query: userMsg.content,
                use_hybrid: false
            });

            const botMsg = {
                role: 'assistant',
                content: res.data.answer,
                citations: res.data.citations
            };
            setMessages((prev) => [...prev, botMsg]);
        } catch (err) {
            setMessages((prev) => [
                ...prev,
                { role: 'assistant', content: 'Error communicating with generation backend service.' }
            ]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full bg-gray-900 text-gray-100 p-4 rounded-xl shadow-xl">
            <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2">
                {messages.map((msg, idx) => (
                    <div
                        key={idx}
                        className={`p-4 rounded-lg max-w-2xl ${
                            msg.role === 'user' ? 'ml-auto bg-blue-600 text-white' : 'mr-auto bg-gray-800 border border-gray-700'
                        }`}
                    >
                        <p className="whitespace-pre-wrap">{msg.content}</p>
                        {msg.citations && msg.citations.length > 0 && (
                            <div className="mt-3 pt-3 border-t border-gray-700 flex flex-wrap gap-2">
                                <span className="text-xs text-gray-400 font-semibold self-center">Sources:</span>
                                {msg.citations.map((cite, cIdx) => (
                                    <button
                                        key={cIdx}
                                        onClick={() => onSelectCitation(cite)}
                                        className="text-xs bg-gray-700 hover:bg-blue-500 text-gray-200 px-2 py-1 rounded transition flex items-center space-x-1"
                                    >
                                        <span>{cite.source_doc}</span>
                                        {cite.page_number && <span>(p. {cite.page_number})</span>}
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>
                ))}
                {loading && <div className="text-gray-400 italic">Thinking and fetching context...</div>}
                <div ref={messagesEndRef} />
            </div>

            <form onSubmit={handleSendMessage} className="flex gap-2">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask a question regarding your documents..."
                    className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:outline-none focus:border-blue-500"
                />
                <button
                    type="submit"
                    className="bg-blue-600 hover:bg-blue-500 px-6 py-2 rounded-lg font-semibold transition"
                >
                    Send
                </button>
            </form>
        </div>
    );
}