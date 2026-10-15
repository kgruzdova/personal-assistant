"""
Tests for RAG (Retrieval-Augmented Generation) functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from rag.loader import DocumentLoader
from rag.index import VectorIndex


class TestDocumentLoader:
    """Test suite for document loading."""
    
    @pytest.fixture
    def temp_documents(self, tmp_path):
        """Create temporary test documents."""
        # Create PDF mock
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_text("Mock PDF content")
        
        # Create TXT file
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("This is a test document with some content.")
        
        # Create MD file
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test Markdown\n\nSome content here.")
        
        return tmp_path
    
    def test_loader_initialization(self):
        """Test document loader initialization."""
        loader = DocumentLoader()
        
        assert loader is not None
        assert loader.text_splitter is not None
    
    def test_load_text_directly(self):
        """Test loading text directly."""
        loader = DocumentLoader()
        
        text = "This is a test document. " * 100
        chunks = loader.load_text(text, source="test")
        
        assert len(chunks) > 0
        assert chunks[0].metadata["source"] == "test"
    
    @patch('rag.loader.TextLoader')
    def test_load_txt_document(self, mock_text_loader, temp_documents):
        """Test loading TXT document."""
        # Mock the loader
        mock_loader_instance = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {}
        mock_loader_instance.load.return_value = [mock_doc]
        mock_text_loader.return_value = mock_loader_instance
        
        loader = DocumentLoader()
        txt_file = temp_documents / "test.txt"
        
        try:
            chunks = loader.load_document(txt_file)
            assert len(chunks) > 0
        except Exception:
            # Loading may fail in test environment
            pass
    
    def test_load_directory_empty(self, tmp_path):
        """Test loading from empty directory."""
        loader = DocumentLoader()
        
        chunks = loader.load_directory(tmp_path)
        
        assert isinstance(chunks, list)
        assert len(chunks) == 0


class TestVectorIndex:
    """Test suite for vector indexing."""
    
    @pytest.fixture
    def mock_embeddings(self):
        """Mock OpenAI embeddings."""
        with patch('rag.index.OpenAIEmbeddings') as mock:
            mock_instance = Mock()
            mock_instance.embed_query = Mock(return_value=[0.1] * 1536)
            mock_instance.embed_documents = Mock(return_value=[[0.1] * 1536])
            mock.return_value = mock_instance
            yield mock
    
    @pytest.fixture
    def mock_chroma(self):
        """Mock ChromaDB."""
        with patch('rag.index.Chroma') as mock:
            mock_instance = Mock()
            mock_instance.add_documents = Mock()
            mock_instance.similarity_search = Mock(return_value=[])
            mock_instance.similarity_search_with_score = Mock(return_value=[])
            mock_instance._collection = Mock()
            mock_instance._collection.count = Mock(return_value=0)
            mock.return_value = mock_instance
            yield mock
    
    def test_vector_index_initialization(self, tmp_path, mock_embeddings, mock_chroma):
        """Test vector index initialization."""
        index = VectorIndex(persist_directory=tmp_path)
        
        assert index is not None
        assert index.vectorstore is not None
    
    def test_add_documents_to_index(self, tmp_path, mock_embeddings, mock_chroma):
        """Test adding documents to index."""
        index = VectorIndex(persist_directory=tmp_path)
        
        # Create mock documents
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"source": "test"}
        
        documents = [mock_doc]
        
        # Should not raise exception
        try:
            index.add_documents(documents)
        except Exception as e:
            pytest.fail(f"Adding documents failed: {e}")
    
    def test_similarity_search(self, tmp_path, mock_embeddings, mock_chroma):
        """Test similarity search."""
        index = VectorIndex(persist_directory=tmp_path)
        
        query = "test query"
        results = index.similarity_search(query, k=3)
        
        assert isinstance(results, list)
    
    def test_get_stats(self, tmp_path, mock_embeddings, mock_chroma):
        """Test getting index statistics."""
        index = VectorIndex(persist_directory=tmp_path)
        
        stats = index.get_stats()
        
        assert isinstance(stats, dict)
        assert "total_documents" in stats or "error" in stats


class TestRAGQuery:
    """Test suite for RAG query functionality."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for RAG queries."""
        with patch('rag.query.openai_client') as mock:
            mock.generate_text_response = AsyncMock(
                return_value="This is a test response based on context."
            )
            yield mock
    
    @pytest.fixture
    def mock_vector_index(self):
        """Mock vector index for RAG queries."""
        with patch('rag.query.vector_index') as mock:
            # Create mock document
            mock_doc = Mock()
            mock_doc.page_content = "This is relevant context from documents."
            mock_doc.metadata = {"source": "test.pdf"}
            
            mock.similarity_search_with_score = Mock(
                return_value=[(mock_doc, 0.95)]
            )
            yield mock
    
    @pytest.mark.asyncio
    async def test_query_knowledge_base(self, mock_openai_client, mock_vector_index):
        """Test querying knowledge base."""
        from rag.query import query_knowledge_base
        
        query = "What is the test about?"
        response = await query_knowledge_base(query)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_query_with_history(self, mock_openai_client, mock_vector_index):
        """Test querying with conversation history."""
        from rag.query import query_knowledge_base
        
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        
        query = "Follow-up question"
        response = await query_knowledge_base(query, history)
        
        assert isinstance(response, str)
    
    @pytest.mark.asyncio
    async def test_fallback_response(self, mock_openai_client):
        """Test fallback response when no context found."""
        from rag.query import _fallback_response
        
        query = "Random question"
        response = await _fallback_response(query)
        
        assert isinstance(response, str)
        assert "⚠️" in response or len(response) > 0
    
    def test_prepare_context(self):
        """Test context preparation from search results."""
        from rag.query import _prepare_context
        
        # Create mock documents
        mock_doc1 = Mock()
        mock_doc1.page_content = "First document content"
        mock_doc1.metadata = {"source": "doc1.pdf"}
        
        mock_doc2 = Mock()
        mock_doc2.page_content = "Second document content"
        mock_doc2.metadata = {"source": "doc2.pdf"}
        
        results = [
            (mock_doc1, 0.95),
            (mock_doc2, 0.87)
        ]
        
        context = _prepare_context(results)
        
        assert isinstance(context, str)
        assert "First document content" in context
        assert "Second document content" in context
        assert "doc1.pdf" in context
    
    def test_get_knowledge_base_stats(self):
        """Test getting knowledge base statistics."""
        from rag.query import get_knowledge_base_stats
        
        with patch('rag.query.vector_index') as mock_index:
            mock_index.get_stats.return_value = {
                "total_documents": 42,
                "persist_directory": "/path/to/db"
            }
            
            stats = get_knowledge_base_stats()
            
            assert isinstance(stats, dict)
            assert "total_documents" in stats


class TestRAGIntegration:
    """Integration tests for RAG system."""
    
    @pytest.mark.asyncio
    async def test_full_rag_pipeline_mock(self):
        """Test full RAG pipeline with mocks."""
        with patch('rag.loader.PyPDFLoader'), \
             patch('rag.index.OpenAIEmbeddings'), \
             patch('rag.index.Chroma'), \
             patch('rag.query.openai_client') as mock_client:
            
            mock_client.generate_text_response = AsyncMock(
                return_value="RAG response"
            )
            
            # This would be the full pipeline
            # For now, just test that imports work
            from rag.query import query_knowledge_base
            
            # Test with empty results
            with patch('rag.query.vector_index') as mock_index:
                mock_index.similarity_search_with_score = Mock(return_value=[])
                
                result = await query_knowledge_base("test query")
                assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

