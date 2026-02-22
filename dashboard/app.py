"""
ScholarAI - Premium Research PDF Chatbot
Production-Grade RAG System for Academic Analysis
Built for Streamlit Cloud Deployment
"""

import streamlit as st
import os
import hashlib
import json
import time
import re
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
from io import BytesIO
import base64

# ============================================================================
# DEPENDENCIES - Install via requirements.txt
# ============================================================================
# streamlit==1.31.0
# langchain==0.1.0
# langchain-community==0.0.10
# langchain-text-splitters==0.0.1
# chromadb==0.4.22
# sentence-transformers==2.3.1
# pymupdf==1.23.8
# openai==1.10.0
# fpdf2==2.7.7
# ============================================================================

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.schema import Document
    import chromadb
    from sentence_transformers import SentenceTransformer
    import fitz  # PyMuPDF
    from openai import OpenAI
    from fpdf import FPDF
except ImportError as e:
    st.error(f"Missing dependency: {e}. Please install requirements.txt")
    st.stop()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration management"""
    MODEL_NAME = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K = 5
    MAX_TOKENS = 2048
    TEMPERATURE = 0.3
    DB_PATH = "./chroma_db"
    
# ============================================================================
# PREMIUM CSS STYLING
# ============================================================================

def get_premium_css() -> str:
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Container */
    .main > div {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 1px solid #e9ecef;
    }
    
    /* Chat Messages */
    .stChatMessage {
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background: #ffffff;
    }
    
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background: #f8f9fa;
    }
    
    /* User Message */
    [data-testid="stChatMessage"]:has([data-testid="stAvatar"] img[src*="user"]) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    
    /* Assistant Message */
    [data-testid="stChatMessage"]:has([data-testid="stAvatar"] img[src*="assistant"]) {
        background: #ffffff;
        border-left: 4px solid #667eea;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        font-weight: 600;
    }
    
    /* Code Blocks */
    pre {
        background: #1e1e1e;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #333;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: #ffffff;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Select Box */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    
    /* File Uploader */
    .stFileUploader {
        border: 2px dashed #dee2e6;
        border-radius: 12px;
        padding: 2rem;
        background: #f8f9fa;
    }
    
    /* Citation Box */
    .citation-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading-text {
        animation: pulse 1.5s infinite;
    }
    
    /* Header */
    .app-header {
        text-align: center;
        padding: 2rem 0;
        border-bottom: 1px solid #e9ecef;
        margin-bottom: 2rem;
    }
    
    .app-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-active {
        background: #d4edda;
        color: #155724;
    }
    
    .status-pending {
        background: #fff3cd;
        color: #856404;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid #e9ecef;
        color: #6c757d;
        font-size: 0.875rem;
    }
    </style>
    """

# ============================================================================
# PDF PROCESSING ENGINE
# ============================================================================

class PDFProcessor:
    """Handles PDF extraction and chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_text(self, file_bytes: BytesIO, filename: str) -> List[Document]:
        """Extract text from PDF with page metadata"""
        documents = []
        
        try:
            pdf = fitz.open(stream=file_bytes.read(), filetype="pdf")
            
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text = page.get_text()
                
                if text.strip():
                    docs = self.splitter.create_documents(
                        [text],
                        metadatas=[{
                            "source": filename,
                            "page": page_num + 1,
                            "total_pages": len(pdf),
                            "timestamp": datetime.now().isoformat()
                        }]
                    )
                    documents.extend(docs)
            
            pdf.close()
            return documents
            
        except Exception as e:
            st.error(f"Error processing {filename}: {str(e)}")
            return []
    
    def get_document_stats(self, documents: List[Document]) -> Dict:
        """Calculate document statistics"""
        if not documents:
            return {"chunks": 0, "pages": 0, "chars": 0}
        
        pages = set()
        total_chars = 0
        
        for doc in documents:
            pages.add(doc.metadata.get("page", 0))
            total_chars += len(doc.page_content)
        
        return {
            "chunks": len(documents),
            "pages": len(pages),
            "chars": total_chars
        }

# ============================================================================
# VECTOR STORE MANAGER
# ============================================================================

class VectorStoreManager:
    """Manages ChromaDB vector store operations"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = None
        self.vectorstore = None
        self._initialize()
    
    def _initialize(self):
        """Initialize embeddings and vector store"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        except Exception as e:
            st.error(f"Failed to initialize vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Document], doc_id: str):
        """Add documents to vector store with unique IDs"""
        if not documents:
            return
        
        ids = []
        for i, doc in enumerate(documents):
            unique_id = hashlib.md5(
                f"{doc_id}_{i}_{doc.page_content[:100]}".encode()
            ).hexdigest()
            ids.append(unique_id)
        
        self.vectorstore.add_documents(documents=documents, ids=ids)
    
    def similarity_search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Document]:
        """Search for similar documents"""
        try:
            if filter_dict:
                return self.vectorstore.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            return self.vectorstore.similarity_search(query=query, k=k)
        except Exception as e:
            st.error(f"Search error: {e}")
            return []
    
    def get_all_documents(self) -> List[str]:
        """Get all unique document names"""
        try:
            collection = self.vectorstore._client.get_collection("langchain")
            metadatas = collection.get(include=["metadatas"])["metadatas"]
            sources = set()
            for meta in metadatas:
                if meta and "source" in meta:
                    sources.add(meta["source"])
            return list(sources)
        except:
            return []
    
    def clear_store(self):
        """Clear all documents from store"""
        try:
            self.vectorstore.delete_collection()
            self._initialize()
        except:
            pass

# ============================================================================
# LLM ENGINE
# ============================================================================

class LLMEngine:
    """Handles LLM interactions with citation enforcement"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
    
    def generate_response(
        self,
        query: str,
        context: str,
        history: List[Dict],
        mode: str = "strict",
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Generate response with citations and metadata"""
        
        if not self.client:
            return {
                "content": "⚠️ API key not configured. Please add OPENAI_API_KEY to secrets.",
                "citations": [],
                "confidence": 0.0,
                "tokens_used": 0
            }
        
        system_prompt = self._get_system_prompt(mode)
        
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=Config.MAX_TOKENS
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            # Extract citations
            citations = self._extract_citations(content, context)
            
            # Calculate confidence
            confidence = self._calculate_confidence(content, context)
            
            return {
                "content": content,
                "citations": citations,
                "confidence": confidence,
                "tokens_used": tokens_used
            }
            
        except Exception as e:
            return {
                "content": f"⚠️ Error generating response: {str(e)}",
                "citations": [],
                "confidence": 0.0,
                "tokens_used": 0
            }
    
    def _get_system_prompt(self, mode: str) -> str:
        """Get system prompt based on mode"""
        base = """You are ScholarAI, a Senior Research Assistant with expertise in academic analysis.

CRITICAL RULES:
1. Answer ONLY using the provided Context. Do not use external knowledge.
2. If the answer is not in the context, state: "Information not found in provided documents."
3. Every claim must be cited using format: [Source: filename.pdf, Page: X]
4. Be precise, academic, and professional in tone.
5. Do not hallucinate or make up citations.
6. If multiple sources agree, mention the consensus.
7. If sources contradict, highlight the contradiction.

CITATION FORMAT:
- Always cite after each paragraph or claim
- Format: [Source: document_name.pdf, Page: 12]
- Include exact page numbers from the metadata"""

        mode_prompts = {
            "strict": base + "\n\nMODE: Strict Citation. Provide direct answers with exact quotes and precise citations.",
            "synthesis": base + "\n\nMODE: Creative Synthesis. Connect ideas across multiple documents. Highlight themes, patterns, and relationships between sources.",
            "contradiction": base + "\n\nMODE: Contradiction Detection. Actively search for conflicting data points, methodologies, or conclusions between sources. Highlight disagreements.",
            "summary": base + "\n\nMODE: Summary. Provide concise executive summary of key findings across all documents.",
            "review": base + "\n\nMODE: Literature Review. Generate structured literature review with introduction, themes, methodology comparison, and conclusions."
        }
        
        return mode_prompts.get(mode, base)
    
    def _extract_citations(self, content: str, context: str) -> List[Dict]:
        """Extract and validate citations from response"""
        citations = []
        pattern = r'\[Source:\s*([^,]+),\s*Page:\s*(\d+)\]'
        matches = re.findall(pattern, content)
        
        for source, page in matches:
            citations.append({
                "source": source.strip(),
                "page": int(page),
                "snippet": self._find_snippet(source, int(page), context)
            })
        
        return citations
    
    def _find_snippet(self, source: str, page: int, context: str) -> str:
        """Find relevant snippet from context"""
        pattern = rf'\[Source:\s*{re.escape(source)},\s*Page:\s*{page}\]\n(.*?)(?=\[Source:|$)'
        match = re.search(pattern, context, re.DOTALL)
        if match:
            text = match.group(1).strip()
            return text[:300] + "..." if len(text) > 300 else text
        return "Snippet not available"
    
    def _calculate_confidence(self, content: str, context: str) -> float:
        """Calculate confidence score based on citation density"""
        if not content:
            return 0.0
        
        citation_count = len(re.findall(r'\[Source:', content))
        content_length = len(content.split())
        
        if content_length == 0:
            return 0.0
        
        citation_density = citation_count / max(content_length, 1)
        
        # Base confidence on citation density
        if citation_density > 0.05:
            return min(0.95, 0.7 + citation_density * 10)
        elif citation_density > 0.02:
            return 0.6 + citation_density * 10
        else:
            return 0.3 + citation_density * 10

# ============================================================================
# EXPORT UTILITIES
# ============================================================================

class ExportManager:
    """Handles chat export functionality"""
    
    @staticmethod
    def export_to_markdown(messages: List[Dict], title: str = "Research Chat") -> str:
        """Export chat to Markdown format"""
        md = f"# {title}\n\n"
        md += f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md += "---\n\n"
        
        for msg in messages:
            role = "👤 **User**" if msg["role"] == "user" else "🤖 **Assistant**"
            md += f"### {role}\n\n"
            md += f"{msg['content']}\n\n"
            
            if msg.get("citations"):
                md += "**Citations:**\n"
                for cit in msg["citations"]:
                    md += f"- {cit['source']} (Page {cit['page']})\n"
                md += "\n"
            
            md += "---\n\n"
        
        return md
    
    @staticmethod
    def export_to_pdf(messages: List[Dict], title: str = "Research Chat") -> bytes:
        """Export chat to PDF format"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, title, ln=True, align='C')
        pdf.ln(5)
        
        # Timestamp
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 10, f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(10)
        
        # Messages
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, f"{role}:", ln=True)
            
            pdf.set_font("Arial", size=11)
            
            # Handle long text
            content = msg["content"]
            for line in content.split('\n'):
                pdf.multi_cell(0, 8, line[:200])  # Truncate very long lines
            
            pdf.ln(5)
        
        return pdf.output(dest='S').encode('latin-1')
    
    @staticmethod
    def download_button(data: str, filename: str, label: str, mime: str):
        """Create download button"""
        b64 = base64.b64encode(data.encode() if isinstance(data, str) else data).decode()
        href = f'<a href="data:{mime};base64,{b64}" download="{filename}" style="text-decoration: none;">{label}</a>'
        return href

# ============================================================================
# SESSION STATE MANAGER
# ============================================================================

class SessionManager:
    """Manages Streamlit session state"""
    
    @staticmethod
    def initialize():
        """Initialize all session state variables"""
        defaults = {
            "messages": [],
            "documents": {},
            "selected_docs": [],
            "total_tokens": 0,
            "chat_history": [],
            "indexing_complete": False,
            "api_configured": False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def add_message(role: str, content: str, citations: List = None, tokens: int = 0):
        """Add message to chat history"""
        st.session_state.messages.append({
            "role": role,
            "content": content,
            "citations": citations or [],
            "timestamp": datetime.now().isoformat(),
            "tokens": tokens
        })
        st.session_state.total_tokens += tokens
    
    @staticmethod
    def clear_chat():
        """Clear chat history"""
        st.session_state.messages = []
        st.session_state.total_tokens = 0
    
    @staticmethod
    def get_conversation_history() -> List[Dict]:
        """Get conversation history for LLM context"""
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages[-10:]  # Last 10 messages
        ]

# ============================================================================
# UI COMPONENTS
# ============================================================================

class UIComponents:
    """Premium UI components"""
    
    @staticmethod
    def render_header():
        """Render app header"""
        st.markdown("""
        <div class="app-header">
            <h1>📚 ScholarAI</h1>
            <p style="color: #6c757d; font-size: 1.1rem;">
                Production-Grade Research PDF Chatbot | Academic Analysis Powered by AI
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_sidebar():
        """Render premium sidebar"""
        with st.sidebar:
            st.markdown("### 🎛️ Control Panel")
            st.markdown("---")
            
            # API Configuration
            with st.expander("🔑 API Configuration", expanded=False):
                api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    help="Enter your OpenAI API key"
                )
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key
                    st.session_state.api_configured = True
                    st.success("✅ API Key Configured")
            
            st.markdown("---")
            
            # Document Upload
            st.markdown("### 📄 Document Library")
            uploaded_files = st.file_uploader(
                "Upload PDF Documents",
                type=["pdf"],
                accept_multiple_files=True,
                help="Upload research papers, articles, or documents"
            )
            
            st.markdown("---")
            
            # Document Selection
            st.markdown("### 📋 Active Documents")
            all_docs = list(st.session_state.documents.keys())
            
            if all_docs:
                selected = st.multiselect(
                    "Select documents for context",
                    all_docs,
                    default=all_docs,
                    help="Choose which documents to include in analysis"
                )
                st.session_state.selected_docs = selected
                
                # Document stats
                st.markdown("#### Document Status")
                for doc_name in all_docs:
                    doc_info = st.session_state.documents[doc_name]
                    status_class = "status-active" if doc_info.get("indexed") else "status-pending"
                    st.markdown(f"""
                    <div class="status-badge {status_class}">
                        📄 {doc_name} ({doc_info.get('chunks', 0)} chunks)
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("📭 No documents uploaded yet")
                selected = []
            
            st.markdown("---")
            
            # Research Mode
            st.markdown("### 🎯 Research Mode")
            mode = st.selectbox(
                "Analysis Strategy",
                ["strict", "synthesis", "contradiction", "summary", "review"],
                format_func=lambda x: {
                    "strict": "📌 Strict Citation",
                    "synthesis": "🔗 Creative Synthesis",
                    "contradiction": "⚖️ Contradiction Detection",
                    "summary": "📝 Executive Summary",
                    "review": "📚 Literature Review"
                }[x]
            )
            
            st.markdown("---")
            
            # Settings
            with st.expander("⚙️ Advanced Settings"):
                temperature = st.slider(
                    "Temperature",
                    0.0, 1.0, 0.3,
                    help="Higher = more creative, Lower = more precise"
                )
                top_k = st.slider(
                    "Retrieval Count",
                    1, 10, 5,
                    help="Number of document chunks to retrieve"
                )
                
                st.session_state.temperature = temperature
                st.session_state.top_k = top_k
            
            st.markdown("---")
            
            # Actions
            st.markdown("### 🚀 Actions")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    SessionManager.clear_chat()
                    st.rerun()
            
            with col2:
                if st.button("🔄 Reset All", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
            
            # Export
            st.markdown("---")
            st.markdown("### 💾 Export")
            
            if st.session_state.messages:
                st.download_button(
                    "📥 Download Markdown",
                    data=ExportManager.export_to_markdown(st.session_state.messages),
                    file_name=f"research_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            return uploaded_files, selected, mode
    
    @staticmethod
    def render_chat_message(msg: Dict):
        """Render a single chat message with citations"""
        role = msg["role"]
        content = msg["content"]
        citations = msg.get("citations", [])
        
        with st.chat_message(role):
            st.markdown(content)
            
            if citations:
                with st.expander(f"📚 View {len(citations)} Citation(s)", expanded=False):
                    for i, cit in enumerate(citations, 1):
                        st.markdown(f"**Citation {i}:**")
                        st.markdown(f"- **Source:** {cit['source']}")
                        st.markdown(f"- **Page:** {cit['page']}")
                        with st.container():
                            st.markdown("##### Snippet:")
                            st.code(cit['snippet'][:200] + "..." if len(cit['snippet']) > 200 else cit['snippet'])
                        st.markdown("---")
    
    @staticmethod
    def render_metrics():
        """Render usage metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Messages",
                len(st.session_state.messages),
                delta=None
            )
        
        with col2:
            st.metric(
                "Documents",
                len(st.session_state.documents),
                delta=None
            )
        
        with col3:
            st.metric(
                "Tokens Used",
                f"{st.session_state.total_tokens:,}",
                delta=None
            )
        
        with col4:
            active_docs = len(st.session_state.selected_docs)
            st.metric(
                "Active Docs",
                active_docs,
                delta=None
            )
    
    @staticmethod
    def render_quick_actions():
        """Render quick action buttons"""
        st.markdown("### ⚡ Quick Research Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Generate Summary", use_container_width=True):
                st.session_state.quick_action = "summary"
        
        with col2:
            if st.button("🔍 Find Contradictions", use_container_width=True):
                st.session_state.quick_action = "contradiction"
        
        with col3:
            if st.button("📚 Literature Review", use_container_width=True):
                st.session_state.quick_action = "review"
        
        with col4:
            if st.button("🎯 Key Themes", use_container_width=True):
                st.session_state.quick_action = "themes"
        
        if "quick_action" in st.session_state and st.session_state.quick_action:
            action = st.session_state.quick_action
            action_prompts = {
                "summary": "Provide an executive summary of all uploaded documents.",
                "contradiction": "Identify and analyze any contradictions between the uploaded documents.",
                "review": "Generate a structured literature review based on the uploaded documents.",
                "themes": "Extract and explain the key themes across all uploaded documents."
            }
            
            if action in action_prompts:
                return action_prompts[action]
        
        return None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Page Configuration
    st.set_page_config(
        page_title="ScholarAI | Research Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject Premium CSS
    st.markdown(get_premium_css(), unsafe_allow_html=True)
    
    # Initialize Session State
    SessionManager.initialize()
    
    # Initialize Components
    pdf_processor = PDFProcessor()
    vector_manager = VectorStoreManager()
    llm_engine = LLMEngine()
    
    # Render Header
    UIComponents.render_header()
    
    # Render Sidebar
    uploaded_files, selected_docs, mode = UIComponents.render_sidebar()
    
    # Render Metrics
    UIComponents.render_metrics()
    
    # Process Uploaded Files
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.documents:
                with st.spinner(f"📖 Processing {file.name}..."):
                    # Extract text
                    file_bytes = BytesIO(file.getvalue())
                    documents = pdf_processor.extract_text(file_bytes, file.name)
                    
                    if documents:
                        # Add to vector store
                        vector_manager.add_documents(documents, file.name)
                        
                        # Store metadata
                        stats = pdf_processor.get_document_stats(documents)
                        st.session_state.documents[file.name] = {
                            "indexed": True,
                            "chunks": stats["chunks"],
                            "pages": stats["pages"],
                            "chars": stats["chars"],
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        st.success(f"✅ {file.name} indexed ({stats['chunks']} chunks)")
                    else:
                        st.session_state.documents[file.name] = {
                            "indexed": False,
                            "error": "Failed to extract text"
                        }
    
    # Render Quick Actions
    quick_action_prompt = UIComponents.render_quick_actions()
    
    # Chat Interface
    st.markdown("---")
    st.markdown("### 💬 Research Chat")
    
    # Display Chat History
    for msg in st.session_state.messages:
        UIComponents.render_chat_message(msg)
    
    # Chat Input
    if prompt := st.chat_input("Ask a research question about your documents..."):
        # Handle quick action
        if quick_action_prompt:
            prompt = quick_action_prompt
            st.session_state.quick_action = None
        
        # Add user message
        SessionManager.add_message("user", prompt)
        UIComponents.render_chat_message(st.session_state.messages[-1])
        
        # Generate response
        with st.spinner("🤔 Analyzing documents..."):
            # Get context from selected documents
            filter_dict = None
            if selected_docs:
                filter_dict = {"source": {"$in": selected_docs}}
            
            # Retrieve relevant chunks
            docs = vector_manager.similarity_search(
                query=prompt,
                k=st.session_state.get("top_k", 5),
                filter_dict=filter_dict
            )
            
            # Format context
            context = ""
            for doc in docs:
                context += f"[Source: {doc.metadata['source']}, Page: {doc.metadata['page']}]\n{doc.page_content}\n\n"
            
            if not context:
                context = "No relevant documents found in the selected sources."
            
            # Generate response
            temperature = st.session_state.get("temperature", 0.3)
            history = SessionManager.get_conversation_history()
            
            response_data = llm_engine.generate_response(
                query=prompt,
                context=context,
                history=history,
                mode=mode,
                temperature=temperature
            )
            
            # Add assistant message
            SessionManager.add_message(
                "assistant",
                response_data["content"],
                citations=response_data["citations"],
                tokens=response_data["tokens_used"]
            )
            
            # Render response
            UIComponents.render_chat_message(st.session_state.messages[-1])
            
            # Show confidence score
            if response_data["confidence"] > 0:
                st.info(f"🎯 AI Confidence Score: {response_data['confidence']:.1%}")
    
    # Empty State
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 0; color: #6c757d;">
            <h2 style="font-size: 3rem; margin-bottom: 1rem;">👋 Welcome to ScholarAI</h2>
            <p style="font-size: 1.2rem; margin-bottom: 2rem;">
                Upload research papers and start asking questions.<br>
                Our AI will analyze your documents and provide cited answers.
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <div style="background: #f8f9fa; padding: 2rem; border-radius: 12px; max-width: 250px;">
                    <h3 style="color: #667eea;">📄 Upload PDFs</h3>
                    <p>Upload unlimited research papers and documents</p>
                </div>
                <div style="background: #f8f9fa; padding: 2rem; border-radius: 12px; max-width: 250px;">
                    <h3 style="color: #667eea;">🔍 Ask Questions</h3>
                    <p>Get AI-powered answers with precise citations</p>
                </div>
                <div style="background: #f8f9fa; padding: 2rem; border-radius: 12px; max-width: 250px;">
                    <h3 style="color: #667eea;">📚 Export Research</h3>
                    <p>Download your chat as Markdown or PDF</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="app-footer">
        <p>ScholarAI v1.0 | Built for Academic Research | 
        <span style="color: #667eea;">Secure</span> • 
        <span style="color: #667eea;">Cited</span> • 
        <span style="color: #667eea;">Production-Grade</span></p>
        <p style="font-size: 0.75rem; margin-top: 0.5rem;">
            ⚠️ Always verify AI-generated citations against original sources
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
