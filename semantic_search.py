#!/usr/bin/env python3
"""
Semantic Search Engine with Flexible Document Management

A semantic search engine with embedding model and cross-encoder reranking.

Features:
- Flexible document management (add one at a time or in batches)
- Optional cross-encoder reranking for improved accuracy
- Automatic or manual indexing control
- Document persistence (save/load)
- Method chaining for fluent interface

Memory Usage (for all-MiniLM-L12-v2):
- Model: ~200 MB (embedding + cross-encoder models)
- 1K documents: ~1.5 MB for embeddings (~201.5 MB total)
- 10K documents: ~15 MB for embeddings (~215 MB total)
- 100K documents: ~150 MB for embeddings (~350 MB total)

Speed (CPU on M-series Mac):
- CPU encoding: ~550 docs/sec (varies with document length)
- Search latency: ~40ms (includes cross-encoder reranking)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
from dataclasses import dataclass
import numpy as np
import time
import pickle
from pathlib import Path
from sentence_transformers import CrossEncoder
import torch
from embedding_providers import create_embedding_provider


@dataclass
class SearchResult:
    """Container for search results"""

    score: float
    text: str
    index: int


class SemanticSearchInterface(ABC):
    """
    Abstract interface for semantic search implementations.
    Defines the contract that all semantic search engines must follow.
    """

    @abstractmethod
    def add(self, documents: Union[str, List[str]]) -> "SemanticSearchInterface":
        """Add one or more documents to the index."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10, **_kwargs) -> List[SearchResult]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def remove(self, indices: Union[int, List[int]]) -> "SemanticSearchInterface":
        """Remove documents by index."""
        pass

    @abstractmethod
    def clear(self) -> "SemanticSearchInterface":
        """Clear all documents from the index."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return the number of indexed documents."""
        pass

    @abstractmethod
    def get_documents(self) -> List[str]:
        """Return all indexed documents."""
        pass


class SearchConfig:
    """Configuration for the search engine"""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L12-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: Optional[str] = None,
        use_half_precision: bool = False,
        cache_dir: Optional[str] = "./cache",
        auto_index: bool = True,
        batch_size: int = 32,
        bedrock_region: Optional[str] = None,
        embedding_dimensions: Optional[int] = None,
    ):
        """
        Initialize search configuration.

        Args:
            embedding_model: Name of the embedding model. Can be:
                - "model_name" or "local:model_name" for sentence-transformers models
                - "bedrock:model_id" for AWS Bedrock models
            cross_encoder_model: Name of the cross-encoder model for reranking
            device: 'cuda', 'cpu', or None for auto-detection
            use_half_precision: Use float16 to reduce memory (GPU only)
            cache_dir: Directory for caching embeddings
            auto_index: Automatically index documents when added
            batch_size: Batch size for encoding documents
            bedrock_region: AWS region for Bedrock models
            embedding_dimensions: Dimensions for models that support variable dimensions
        """
        self.embedding_model = embedding_model
        self.cross_encoder_model = cross_encoder_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half_precision = use_half_precision
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.auto_index = auto_index
        self.batch_size = batch_size
        self.bedrock_region = bedrock_region
        self.embedding_dimensions = embedding_dimensions

        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)


class SemanticSearch(SemanticSearchInterface):
    """
    Semantic search engine with embedding model and cross-encoder reranking.
    """

    def __init__(
        self,
        documents: Optional[Union[str, List[str]]] = None,
        config: Optional[SearchConfig] = None,
    ):
        """
        Initialize the search engine.

        Args:
            documents: Optional initial documents (single string or list)
            config: SearchConfig object (uses defaults if None)

        Examples:
            # Empty initialization
            searcher = SemanticSearch()

            # With initial documents
            searcher = SemanticSearch(["doc1", "doc2"])

            # With custom config
            config = SearchConfig(auto_index=False)
            searcher = SemanticSearch(config=config)
        """
        self.config = config or SearchConfig()

        # Initialize models
        self._init_models()

        # Document storage
        self._documents: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._is_indexed = False
        self._pending_documents: List[str] = []  # Documents waiting to be indexed

        # Add initial documents if provided
        if documents:
            self.add(documents)

    def _init_models(self):
        """Initialize the ML models"""
        print("Initializing semantic search...")

        # Create embedding provider based on model specification
        self._encoder = create_embedding_provider(
            model_spec=self.config.embedding_model,
            device=self.config.device,
            region_name=self.config.bedrock_region,
            dimensions=self.config.embedding_dimensions,
        )

        # Get model info for logging
        model_info = self._encoder.get_model_info()
        print(f"✓ Embedding provider: {model_info.provider}")
        print(f"✓ Model: {model_info.model_id}")
        print(f"✓ Dimensions: {model_info.dimensions}")

        # Load cross-encoder for reranking (still uses sentence-transformers)
        self._cross_encoder = CrossEncoder(
            self.config.cross_encoder_model, device=self.config.device
        )

        print(f"✓ Cross-encoder: {self.config.cross_encoder_model}")

    def add(self, documents: Union[str, List[str]]) -> "SemanticSearch":
        """
        Add one or more documents to the search engine.

        Args:
            documents: Single document string or list of documents

        Returns:
            self for method chaining

        Examples:
            # Add single document
            searcher.add("Document text")

            # Add multiple documents
            searcher.add(["Doc 1", "Doc 2", "Doc 3"])

            # Chain multiple additions
            searcher.add("Doc 1").add(["Doc 2", "Doc 3"]).add("Doc 4")
        """
        # Convert single string to list
        if isinstance(documents, str):
            documents = [documents]

        # Add to document store
        self._documents.extend(documents)
        self._pending_documents.extend(documents)

        print(f"✓ Added {len(documents)} document(s) (total: {len(self._documents)})")

        # Auto-index if configured
        if self.config.auto_index:
            self._index_pending()
        else:
            self._is_indexed = False
            print("  Note: Call index() to update embeddings or set auto_index=True")

        return self

    def _index_pending(self):
        """Index pending documents and update embeddings"""
        if not self._pending_documents:
            return

        print(f"Indexing {len(self._pending_documents)} new document(s)...")
        start_time = time.time()

        # Encode new documents using the embedding provider
        new_embeddings = self._encoder.encode(
            self._pending_documents,
            batch_size=self.config.batch_size,
            show_progress_bar=len(self._pending_documents) > 100,
            normalize=True,
        )

        # Update embeddings
        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        encoding_time = time.time() - start_time
        docs_per_sec = len(self._pending_documents) / encoding_time
        print(f"  Indexed in {encoding_time:.2f}s ({docs_per_sec:.1f} docs/sec)")

        self._pending_documents.clear()
        self._is_indexed = True

    def index(self, force: bool = False) -> "SemanticSearch":
        """
        Manually trigger indexing of all documents.

        Args:
            force: If True, re-index all documents even if already indexed

        Returns:
            self for method chaining

        Example:
            # Normal indexing (only pending documents)
            searcher.index()

            # Force complete re-indexing
            searcher.index(force=True)
        """
        if force or not self._is_indexed or self._pending_documents:
            if force:
                # Re-index everything
                print("Force re-indexing all documents...")
                self._pending_documents = self._documents.copy()
                self._embeddings = None

            self._index_pending()
        else:
            print("Already indexed. Use force=True to re-index.")

        return self

    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = True,
        rerank_top_n: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search for semantically similar documents.

        Args:
            query: Search query text
            top_k: Number of results to return
            rerank: Whether to use cross-encoder reranking for better accuracy
            rerank_top_n: Number of candidates to consider for reranking (default: 3 * top_k)
            min_score: Minimum score threshold for results

        Returns:
            List of SearchResult objects sorted by relevance

        Example:
            # Basic search
            results = searcher.search("machine learning", top_k=5)

            # Fast search without reranking
            results = searcher.search("python programming", rerank=False)

            # Search with minimum score threshold
            results = searcher.search("data science", min_score=0.5)
        """
        # Ensure documents are indexed
        if not self._is_indexed:
            print("Documents not indexed. Indexing now...")
            self.index()

        if self._embeddings is None or len(self._documents) == 0:
            print("No documents to search")
            return []

        start_time = time.time()

        # Encode query using the embedding provider
        query_embedding = self._encoder.encode(query, normalize=True)
        # Ensure it's 1D for dot product
        if len(query_embedding.shape) > 1:
            query_embedding = query_embedding[0]

        # Calculate similarities (dot product since embeddings are normalized)
        similarities = np.dot(self._embeddings, query_embedding)

        # Get top candidates
        if rerank and top_k > 0:
            # Get more candidates for reranking to improve final results
            candidates_n = rerank_top_n or min(top_k * 3, len(self._documents))
            candidates_n = min(candidates_n, len(self._documents))

            # Get top candidates by embedding similarity
            top_indices = np.argpartition(similarities, -candidates_n)[-candidates_n:]
            top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]

            # Rerank with cross-encoder for better accuracy
            candidate_texts = [self._documents[i] for i in top_indices]
            pairs = [[query, text] for text in candidate_texts]

            cross_scores = self._cross_encoder.predict(pairs, show_progress_bar=False)

            # Sort by cross-encoder scores and take top k
            rerank_indices = np.argsort(cross_scores)[::-1][:top_k]

            results = [
                SearchResult(
                    score=float(cross_scores[idx]),
                    text=candidate_texts[idx],
                    index=int(top_indices[idx]),
                )
                for idx in rerank_indices
            ]

            rerank_info = f" (reranked from {candidates_n} candidates)"
        else:
            # No reranking - use embedding similarities directly
            top_k = min(top_k, len(self._documents))
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]

            results = [
                SearchResult(
                    score=float(similarities[i]),
                    text=self._documents[i],
                    index=int(i),
                )
                for i in top_indices
            ]
            rerank_info = ""

        # Apply minimum score filter if specified
        if min_score is not None:
            original_count = len(results)
            results = [r for r in results if r.score >= min_score]
            if len(results) < original_count:
                print(f"  Filtered to {len(results)} results with score >= {min_score}")

        search_time = time.time() - start_time
        print(f"Search completed in {search_time * 1000:.1f}ms{rerank_info}")

        return results

    def find_similar(
        self, document_index: int, top_k: int = 10, exclude_self: bool = True
    ) -> List[SearchResult]:
        """
        Find documents similar to a document already in the index.

        Args:
            document_index: Index of the reference document
            top_k: Number of similar documents to return
            exclude_self: Whether to exclude the reference document from results

        Returns:
            List of SearchResult objects

        Example:
            # Find documents similar to the first document
            similar = searcher.find_similar(0, top_k=5)
        """
        if not self._is_indexed:
            self.index()

        if self._embeddings is None:
            raise ValueError("No documents indexed")

        if not 0 <= document_index < len(self._documents):
            raise ValueError(f"Invalid document index: {document_index}")

        # Use the document's embedding as query
        query_embedding = self._embeddings[document_index]

        # Calculate similarities
        similarities = np.dot(self._embeddings, query_embedding)

        if exclude_self:
            similarities[document_index] = -1  # Exclude self from results

        # Get top similar documents
        top_k = min(top_k, len(self._documents) - (1 if exclude_self else 0))
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]

        results = [
            SearchResult(
                score=float(similarities[i]),
                text=self._documents[i],
                index=int(i),
            )
            for i in top_indices
            if i != document_index or not exclude_self
        ]

        return results

    def remove(self, indices: Union[int, List[int]]) -> "SemanticSearch":
        """
        Remove documents by index.

        Args:
            indices: Single index or list of indices to remove

        Returns:
            self for method chaining

        Example:
            # Remove single document
            searcher.remove(0)

            # Remove multiple documents
            searcher.remove([1, 3, 5])
        """
        if isinstance(indices, int):
            indices = [indices]

        # Sort indices in reverse order to avoid index shifting issues
        indices = sorted(set(indices), reverse=True)

        removed_count = 0
        for idx in indices:
            if 0 <= idx < len(self._documents):
                del self._documents[idx]
                if self._embeddings is not None:
                    self._embeddings = np.delete(self._embeddings, idx, axis=0)
                removed_count += 1

        print(
            f"✓ Removed {removed_count} document(s) (remaining: {len(self._documents)})"
        )
        return self

    def clear(self) -> "SemanticSearch":
        """
        Clear all documents from the index.

        Returns:
            self for method chaining
        """
        self._documents.clear()
        self._pending_documents.clear()
        self._embeddings = None
        self._is_indexed = False
        print("✓ Cleared all documents")
        return self

    def size(self) -> int:
        """Return the number of indexed documents."""
        return len(self._documents)

    def get_documents(self) -> List[str]:
        """Return a copy of all indexed documents."""
        return self._documents.copy()

    def get_document(self, index: int) -> Optional[str]:
        """
        Get a single document by index.

        Args:
            index: Document index

        Returns:
            Document text or None if index is out of range
        """
        if 0 <= index < len(self._documents):
            return self._documents[index]
        return None

    def save(self, filepath: str) -> "SemanticSearch":
        """
        Save the search index to disk.

        Args:
            filepath: Path to save the index

        Returns:
            self for method chaining

        Example:
            searcher.save("my_index.pkl")
        """
        # Ensure indexed before saving
        if not self._is_indexed:
            self.index()

        # Get model info for saving
        model_info = self._encoder.get_model_info()

        data = {
            "documents": self._documents,
            "embeddings": self._embeddings,
            "config": {
                "embedding_model": self.config.embedding_model,
                "cross_encoder_model": self.config.cross_encoder_model,
                "bedrock_region": self.config.bedrock_region,
                "embedding_dimensions": self.config.embedding_dimensions,
                "model_info": {
                    "provider": model_info.provider,
                    "model_id": model_info.model_id,
                    "dimensions": model_info.dimensions,
                },
            },
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"✓ Saved index to {filepath}")
        return self

    @classmethod
    def load(
        cls, filepath: str, config: Optional[SearchConfig] = None
    ) -> "SemanticSearch":
        """
        Load a search index from disk.

        Args:
            filepath: Path to the saved index
            config: Optional config (uses saved config if None)

        Returns:
            SemanticSearch instance

        Example:
            searcher = SemanticSearch.load("my_index.pkl")
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # Create config if not provided
        if config is None:
            # Preserve original model specification
            saved_config = data.get("config", {})
            config = SearchConfig(
                embedding_model=saved_config.get(
                    "embedding_model", "all-MiniLM-L12-v2"
                ),
                cross_encoder_model=saved_config.get(
                    "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-12-v2"
                ),
                bedrock_region=saved_config.get("bedrock_region"),
                embedding_dimensions=saved_config.get("embedding_dimensions"),
            )

        # Create instance and load data
        instance = cls(config=config)
        instance._documents = data["documents"]
        instance._embeddings = data["embeddings"]
        instance._is_indexed = True

        print(f"✓ Loaded {len(instance._documents)} documents from {filepath}")
        return instance

    # Python special methods for better integration

    def __len__(self) -> int:
        """Support len(searcher) syntax."""
        return self.size()

    def __getitem__(self, index: int) -> str:
        """Support searcher[index] syntax to get a document."""
        doc = self.get_document(index)
        if doc is None:
            raise IndexError(f"Index {index} out of range")
        return doc

    def __contains__(self, document: str) -> bool:
        """Support 'document in searcher' syntax."""
        return document in self._documents

    def __repr__(self) -> str:
        """String representation of the search engine state."""
        return (
            f"SemanticSearch("
            f"documents={len(self._documents)}, "
            f"indexed={self._is_indexed}, "
            f"pending={len(self._pending_documents)})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Semantic Search Engine\n"
            f"  Documents: {len(self._documents)}\n"
            f"  Indexed: {'Yes' if self._is_indexed else 'No'}\n"
            f"  Pending: {len(self._pending_documents)}\n"
            f"  Model: {self.config.embedding_model}\n"
            f"  Device: {self.config.device}"
        )
