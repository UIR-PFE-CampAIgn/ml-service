import asyncio
from typing import AsyncGenerator, Dict, Any, List, Optional
from pathlib import Path
import json

from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, CSVLoader, PDFPlumberLoader
from langchain.schema import Document
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# LLM providers
try:
    from langchain.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from langchain.llms import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from app.core.config import settings
from app.core.logging import ml_logger


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming responses."""
    
    def __init__(self):
        self.tokens = []
        self.finished = False
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Run on new LLM token."""
        self.tokens.append(token)
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Run on LLM end."""
        self.finished = True


class RAGChain:
    """RAG (Retrieval-Augmented Generation) chain using LangChain."""
    
    def __init__(self, 
                 vector_db_path: Optional[str] = None,
                 embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.vector_db_path = vector_db_path or settings.vector_db_path
        self.embeddings_model = embeddings_model
        self.vectorstore = None
        self.retrieval_chain = None
        self.llm = None
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_llm()
        self._load_vectorstore()
    
    def _initialize_embeddings(self):
        """Initialize embeddings model."""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embeddings_model,
                model_kwargs={'device': 'cpu'}  # Use GPU if available: 'cuda'
            )
            ml_logger.info(f"Embeddings initialized with model: {self.embeddings_model}")
        except Exception as e:
            ml_logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def _initialize_llm(self):
        """Initialize LLM based on configuration."""
        try:
            if settings.llm_provider == "ollama" and OLLAMA_AVAILABLE:
                self.llm = Ollama(
                    model=settings.llm_model,
                    base_url=settings.llm_base_url,
                    temperature=0.7
                )
                ml_logger.info(f"Ollama LLM initialized: {settings.llm_model}")
                
            elif settings.llm_provider == "openai" and OPENAI_AVAILABLE:
                self.llm = OpenAI(
                    model_name=settings.llm_model,
                    openai_api_key=settings.llm_api_key,
                    temperature=0.7
                )
                ml_logger.info(f"OpenAI LLM initialized: {settings.llm_model}")
                
            else:
                # Fallback to a simple mock LLM for testing
                from langchain.llms.fake import FakeListLLM
                self.llm = FakeListLLM(responses=[
                    "This is a mock response. Please configure a proper LLM provider."
                ])
                ml_logger.warning("Using mock LLM. Configure Ollama or OpenAI for production use.")
                
        except Exception as e:
            ml_logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _load_vectorstore(self):
        """Load or create vector store."""
        try:
            vectorstore_path = Path(self.vector_db_path)
            
            if vectorstore_path.exists():
                # Load existing vectorstore
                self.vectorstore = Chroma(
                    persist_directory=str(vectorstore_path),
                    embedding_function=self.embeddings
                )
                ml_logger.info(f"Loaded existing vectorstore from {vectorstore_path}")
            else:
                # Create empty vectorstore
                vectorstore_path.mkdir(parents=True, exist_ok=True)
                self.vectorstore = Chroma(
                    persist_directory=str(vectorstore_path),
                    embedding_function=self.embeddings
                )
                ml_logger.info(f"Created new vectorstore at {vectorstore_path}")
                
        except Exception as e:
            ml_logger.error(f"Failed to load vectorstore: {e}")
            raise
    
    def _create_retrieval_chain(self, context_limit: int = 5, temperature: float = 0.7):
        """Create retrieval QA chain."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized")
        
        # Update LLM temperature
        if hasattr(self.llm, 'temperature'):
            self.llm.temperature = temperature
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": context_limit}
        )
        
        # Create QA chain
        self.retrieval_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        return self.retrieval_chain
    
    async def add_documents(self, documents: List[str], metadata_list: List[Dict] = None) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadata_list: Optional list of metadata dicts for each document
        """
        try:
            if not self.vectorstore:
                raise ValueError("Vectorstore not initialized")
            
            # Prepare documents
            doc_objects = []
            for i, doc_text in enumerate(documents):
                metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
                doc_objects.append(Document(page_content=doc_text, metadata=metadata))
            
            # Add to vectorstore
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.vectorstore.add_documents, 
                doc_objects
            )
            
            # Persist
            self.vectorstore.persist()
            
            ml_logger.info(f"Added {len(doc_objects)} documents to vectorstore")
            
        except Exception as e:
            ml_logger.error(f"Failed to add documents: {e}")
            raise
    
    async def load_documents_from_files(self, file_paths: List[str]) -> None:
        """
        Load documents from files and add to vector store.
        
        Args:
            file_paths: List of file paths to load
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            all_documents = []
            
            for file_path in file_paths:
                path = Path(file_path)
                
                if not path.exists():
                    ml_logger.warning(f"File not found: {file_path}")
                    continue
                
                # Choose loader based on file extension
                if path.suffix.lower() == '.pdf':
                    loader = PDFPlumberLoader(str(path))
                elif path.suffix.lower() == '.csv':
                    loader = CSVLoader(str(path))
                else:
                    loader = TextLoader(str(path))
                
                # Load and split documents
                documents = await asyncio.get_event_loop().run_in_executor(
                    None, loader.load
                )
                
                split_docs = await asyncio.get_event_loop().run_in_executor(
                    None, text_splitter.split_documents, documents
                )
                
                all_documents.extend(split_docs)
                ml_logger.info(f"Loaded {len(split_docs)} chunks from {file_path}")
            
            # Add to vectorstore
            if all_documents:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.vectorstore.add_documents,
                    all_documents
                )
                self.vectorstore.persist()
                ml_logger.info(f"Added {len(all_documents)} document chunks to vectorstore")
            
        except Exception as e:
            ml_logger.error(f"Failed to load documents from files: {e}")
            raise
    
    async def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform similarity search in the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        try:
            if not self.vectorstore:
                raise ValueError("Vectorstore not initialized")
            
            # Perform search
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                self.vectorstore.similarity_search_with_score,
                query,
                k
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score)
                })
            
            return formatted_results
            
        except Exception as e:
            ml_logger.error(f"Similarity search failed: {e}")
            raise
    
    async def generate_answer(
        self, 
        query: str, 
        context_limit: int = 5, 
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate answer using RAG.
        
        Args:
            query: User question
            context_limit: Number of context documents to retrieve
            temperature: LLM temperature for generation
            
        Returns:
            Dictionary with answer and source documents
        """
        try:
            # Create retrieval chain
            chain = self._create_retrieval_chain(context_limit, temperature)
            
            # Generate answer
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                chain,
                {"query": query}
            )
            
            # Format response
            response = {
                'query': query,
                'answer': result['result'],
                'source_documents': [
                    {
                        'content': doc.page_content,
                        'metadata': doc.metadata
                    }
                    for doc in result['source_documents']
                ]
            }
            
            ml_logger.debug(f"Generated answer for query: {query}")
            return response
            
        except Exception as e:
            ml_logger.error(f"Answer generation failed: {e}")
            raise
    
    async def stream_answer(
        self, 
        query: str, 
        context_limit: int = 5, 
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming answer using RAG.
        
        Args:
            query: User question
            context_limit: Number of context documents to retrieve
            temperature: LLM temperature for generation
            
        Yields:
            Answer tokens as they are generated
        """
        try:
            # Create callback handler for streaming
            callback_handler = StreamingCallbackHandler()
            
            # Update LLM with streaming callback
            if hasattr(self.llm, 'callbacks'):
                self.llm.callbacks = [callback_handler]
            elif hasattr(self.llm, 'streaming') and hasattr(self.llm, 'callback_manager'):
                self.llm.streaming = True
                from langchain.callbacks.manager import CallbackManager
                self.llm.callback_manager = CallbackManager([callback_handler])
            
            # Create retrieval chain
            chain = self._create_retrieval_chain(context_limit, temperature)
            
            # Start generation in background
            async def run_chain():
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        chain,
                        {"query": query}
                    )
                except Exception as e:
                    ml_logger.error(f"Chain execution failed: {e}")
                    callback_handler.finished = True
            
            # Start the chain
            chain_task = asyncio.create_task(run_chain())
            
            # Stream tokens as they become available
            last_token_count = 0
            
            while not callback_handler.finished:
                current_token_count = len(callback_handler.tokens)
                
                # Yield new tokens
                for i in range(last_token_count, current_token_count):
                    yield callback_handler.tokens[i]
                
                last_token_count = current_token_count
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)
            
            # Wait for chain to complete and yield any remaining tokens
            await chain_task
            
            # Yield any final tokens
            for i in range(last_token_count, len(callback_handler.tokens)):
                yield callback_handler.tokens[i]
                
        except Exception as e:
            ml_logger.error(f"Streaming answer generation failed: {e}")
            yield f"Error: {str(e)}"
    
    def get_vectorstore_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with vectorstore statistics
        """
        try:
            if not self.vectorstore:
                return {'error': 'Vectorstore not initialized'}
            
            # Get collection info
            collection = self.vectorstore._collection
            stats = {
                'document_count': collection.count(),
                'vectorstore_path': self.vector_db_path,
                'embeddings_model': self.embeddings_model
            }
            
            return stats
            
        except Exception as e:
            ml_logger.error(f"Failed to get vectorstore stats: {e}")
            return {'error': str(e)}