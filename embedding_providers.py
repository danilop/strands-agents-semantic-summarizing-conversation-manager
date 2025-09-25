"""
Embedding Provider Abstraction for Semantic Search

This module provides a flexible interface for different embedding providers,
supporting both local sentence-transformers models and AWS Bedrock embedding models.
"""

from typing import Protocol, List, Optional, Union, runtime_checkable
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModelInfo:
    """Information about an embedding model."""

    provider: str
    model_id: str
    dimensions: int
    max_tokens: Optional[int] = None
    supports_normalization: bool = True
    supports_variable_dimensions: bool = False
    available_dimensions: Optional[List[int]] = None


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol defining the embedding provider interface."""

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for processing
            show_progress_bar: Whether to show progress bar
            normalize: Whether to normalize embeddings
            **kwargs: Additional provider-specific parameters

        Returns:
            Numpy array of embeddings
        """
        ...

    def get_model_info(self) -> EmbeddingModelInfo:
        """Get information about the embedding model."""
        ...

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        ...


class LocalEmbeddingProvider:
    """Provider for local sentence-transformers models."""

    def __init__(self, model_name: str, device: Optional[str] = None, **kwargs):
        """
        Initialize local embedding provider.

        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            **kwargs: Additional arguments for SentenceTransformer
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.device = device
        logger.info(f"Loading local embedding model: {model_name}")

        self.model = SentenceTransformer(model_name, device=device, **kwargs)

        # Get embedding dimension
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded with embedding dimension: {self._dimension}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Encode texts using sentence-transformers."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            **kwargs,
        )

        return embeddings

    def get_model_info(self) -> EmbeddingModelInfo:
        """Get information about the local model."""
        return EmbeddingModelInfo(
            provider="local",
            model_id=self.model_name,
            dimensions=self._dimension,
            supports_normalization=True,
            supports_variable_dimensions=False,
        )

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension


class BedrockEmbeddingProvider:
    """Provider for AWS Bedrock embedding models."""

    # Model configurations with their properties
    MODEL_CONFIGS = {
        "amazon.titan-embed-text-v1": {
            "dimensions": 1536,
            "max_tokens": 8000,
            "supports_normalization": True,
            "supports_variable_dimensions": False,
        },
        "amazon.titan-embed-text-v2:0": {
            "dimensions": 1024,  # default
            "max_tokens": 8192,
            "supports_normalization": True,
            "supports_variable_dimensions": True,
            "available_dimensions": [256, 512, 1024],
        },
        "amazon.titan-embed-g1-text-02": {
            "dimensions": 1536,
            "max_tokens": 8000,
            "supports_normalization": True,
            "supports_variable_dimensions": False,
        },
        "cohere.embed-english-v3": {
            "dimensions": 1024,
            "max_tokens": 512,
            "supports_normalization": True,
            "supports_variable_dimensions": False,
        },
        "cohere.embed-multilingual-v3": {
            "dimensions": 1024,
            "max_tokens": 512,
            "supports_normalization": True,
            "supports_variable_dimensions": False,
        },
    }

    def __init__(
        self,
        model_id: str,
        region_name: Optional[str] = None,
        dimensions: Optional[int] = None,
        normalize: bool = True,
        **kwargs,
    ):
        """
        Initialize Bedrock embedding provider.

        Args:
            model_id: Bedrock model ID (e.g., 'amazon.titan-embed-text-v2:0')
            region_name: AWS region name
            dimensions: Embedding dimensions (for models that support variable dimensions)
            normalize: Whether to normalize embeddings by default
            **kwargs: Additional arguments for boto3 client
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for Bedrock embeddings. "
                "Install with: pip install boto3"
            )

        self.model_id = model_id
        self.region_name = region_name

        # Get model configuration
        if model_id not in self.MODEL_CONFIGS:
            logger.warning(
                f"Model {model_id} not in known configurations. Using default settings."
            )
            self.config = {
                "dimensions": dimensions or 1024,
                "max_tokens": 8192,
                "supports_normalization": True,
                "supports_variable_dimensions": False,
            }
        else:
            self.config = self.MODEL_CONFIGS[model_id].copy()

        # Handle variable dimensions
        if dimensions and self.config.get("supports_variable_dimensions"):
            if dimensions in self.config.get("available_dimensions", []):
                self.config["dimensions"] = dimensions
            else:
                raise ValueError(
                    f"Model {model_id} does not support dimension {dimensions}. "
                    f"Available: {self.config.get('available_dimensions')}"
                )
        elif dimensions and not self.config.get("supports_variable_dimensions"):
            logger.warning(
                f"Model {model_id} does not support variable dimensions. "
                f"Using default: {self.config['dimensions']}"
            )

        self._dimension = self.config["dimensions"]

        logger.info(f"Initializing Bedrock client for model: {model_id}")

        # Initialize Bedrock client
        client_args = {"service_name": "bedrock-runtime"}
        if region_name:
            client_args["region_name"] = region_name
        client_args.update(kwargs)

        self.client = boto3.client(**client_args)
        logger.info(f"Bedrock client initialized with dimension: {self._dimension}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Encode texts using AWS Bedrock."""
        import json

        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        # Process in batches (Bedrock typically handles one at a time)
        for i in range(0, len(texts), 1):  # Process one at a time for Bedrock
            text = texts[i]

            # Prepare request body based on model
            if self.model_id.startswith("amazon.titan-embed-text-v2"):
                body = {
                    "inputText": text,
                    "dimensions": self._dimension,
                    "normalize": normalize,
                }
            elif self.model_id.startswith("amazon.titan"):
                # V1 doesn't support these parameters
                body = {"inputText": text}
            elif self.model_id.startswith("cohere"):
                # Cohere has different parameters
                body = {"texts": [text], "input_type": "search_document"}
            else:
                # Default format
                body = {"inputText": text}

            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                )

                response_body = json.loads(response["body"].read())

                # Extract embedding based on model response format
                if "embedding" in response_body:
                    embedding = response_body["embedding"]
                elif "embeddings" in response_body:
                    # Cohere format
                    embedding = response_body["embeddings"][0]
                else:
                    raise ValueError(f"Unexpected response format from {self.model_id}")

                all_embeddings.append(embedding)

            except Exception as e:
                logger.error(f"Error encoding text with Bedrock: {e}")
                raise

            if show_progress_bar and (i + 1) % 10 == 0:
                logger.info(f"Encoded {i + 1}/{len(texts)} texts")

        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # Normalize if requested and not already done by the model
        if normalize and not self.model_id.startswith("amazon.titan-embed-text-v2"):
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / norms

        return embeddings_array

    def get_model_info(self) -> EmbeddingModelInfo:
        """Get information about the Bedrock model."""
        return EmbeddingModelInfo(
            provider="bedrock",
            model_id=self.model_id,
            dimensions=self._dimension,
            max_tokens=self.config.get("max_tokens"),
            supports_normalization=self.config.get("supports_normalization", True),
            supports_variable_dimensions=self.config.get(
                "supports_variable_dimensions", False
            ),
            available_dimensions=self.config.get("available_dimensions"),
        )

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension


def create_embedding_provider(
    model_spec: str,
    device: Optional[str] = None,
    region_name: Optional[str] = None,
    dimensions: Optional[int] = None,
    **kwargs,
) -> EmbeddingProvider:
    """
    Factory function to create an embedding provider based on model specification.

    Args:
        model_spec: Model specification string. Can be:
            - "model_name" or "local:model_name" for local models
            - "bedrock:model_id" for Bedrock models
        device: Device for local models
        region_name: AWS region for Bedrock models
        dimensions: Embedding dimensions for models that support it
        **kwargs: Additional provider-specific arguments

    Returns:
        EmbeddingProvider instance

    Examples:
        >>> provider = create_embedding_provider("all-MiniLM-L12-v2")
        >>> provider = create_embedding_provider("local:all-MiniLM-L12-v2")
        >>> provider = create_embedding_provider("bedrock:amazon.titan-embed-text-v2:0")
    """
    if model_spec.startswith("bedrock:"):
        model_id = model_spec[8:]  # Remove "bedrock:" prefix
        return BedrockEmbeddingProvider(
            model_id=model_id, region_name=region_name, dimensions=dimensions, **kwargs
        )
    elif model_spec.startswith("local:"):
        model_name = model_spec[6:]  # Remove "local:" prefix
        return LocalEmbeddingProvider(model_name=model_name, device=device, **kwargs)
    else:
        # Default to local provider
        return LocalEmbeddingProvider(model_name=model_spec, device=device, **kwargs)
