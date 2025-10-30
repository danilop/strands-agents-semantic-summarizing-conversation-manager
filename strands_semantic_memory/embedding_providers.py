"""Embedding Provider Abstraction for Semantic Search"""

from typing import Protocol, List, Optional, Union, runtime_checkable
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModelInfo:
    provider: str
    model_id: str
    dimensions: int
    max_tokens: Optional[int] = None
    supports_normalization: bool = True
    supports_variable_dimensions: bool = False
    available_dimensions: Optional[List[int]] = None


@runtime_checkable
class EmbeddingProvider(Protocol):
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize: bool = True,
        **kwargs,
    ) -> np.ndarray: ...

    def get_model_info(self) -> EmbeddingModelInfo: ...

    def get_embedding_dimension(self) -> int: ...


class LocalEmbeddingProvider:
    def __init__(self, model_name: str, device: Optional[str] = None, **kwargs):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. Install with: pip install sentence-transformers"
            )
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device, **kwargs)
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
        return EmbeddingModelInfo(
            provider="local",
            model_id=self.model_name,
            dimensions=self._dimension,
            supports_normalization=True,
            supports_variable_dimensions=False,
        )

    def get_embedding_dimension(self) -> int:
        return self._dimension


class BedrockEmbeddingProvider:
    MODEL_CONFIGS = {
        "amazon.nova-2-multimodal-embeddings-v1:0": {
            "dimensions": 3072,
            "max_tokens": 8192,
            "supports_normalization": True,
            "supports_variable_dimensions": True,
            "available_dimensions": [256, 384, 1024, 3072],
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

    MODEL_ALIASES = {
        "nova-multimodal-embeddings": "amazon.nova-2-multimodal-embeddings-v1:0",
    }

    def __init__(
        self,
        model_id: str,
        region_name: Optional[str] = None,
        dimensions: Optional[int] = None,
        normalize: bool = True,
        **kwargs,
    ):
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 is required for Bedrock embeddings. Install with: pip install boto3")

        if model_id in self.MODEL_ALIASES:
            logger.info(f"Resolving model alias '{model_id}' to '{self.MODEL_ALIASES[model_id]}'")
            model_id = self.MODEL_ALIASES[model_id]

        self.model_id = model_id
        self.region_name = region_name
        if model_id not in self.MODEL_CONFIGS:
            logger.warning(f"Model {model_id} not in known configurations. Using default settings.")
            self.config = {
                "dimensions": dimensions or 1024,
                "max_tokens": 8192,
                "supports_normalization": True,
                "supports_variable_dimensions": False,
            }
        else:
            self.config = self.MODEL_CONFIGS[model_id].copy()

        if dimensions and self.config.get("supports_variable_dimensions"):
            if dimensions in self.config.get("available_dimensions", []):
                self.config["dimensions"] = dimensions
            else:
                raise ValueError(
                    f"Model {model_id} does not support dimension {dimensions}. Available: {self.config.get('available_dimensions')}"
                )
        elif dimensions and not self.config.get("supports_variable_dimensions"):
            logger.warning(
                f"Model {model_id} does not support variable dimensions. Using default: {self.config['dimensions']}"
            )

        self._dimension = self.config["dimensions"]
        logger.info(f"Initializing Bedrock client for model: {model_id}")
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
        **_kwargs,
    ) -> np.ndarray:
        import json
        if isinstance(texts, str):
            texts = [texts]
        all_embeddings = []
        for i in range(0, len(texts), 1):
            text = texts[i]
            if self.model_id.startswith("amazon.nova"):
                body = {
                    "taskType": "SINGLE_EMBEDDING",
                    "singleEmbeddingParams": {
                        "embeddingPurpose": "GENERIC_INDEX",
                        "embeddingDimension": self._dimension,
                        "text": {"truncationMode": "END", "value": text},
                    },
                }
            elif self.model_id.startswith("cohere"):
                body = {"texts": [text], "input_type": "search_document"}
            else:
                body = {"inputText": text}
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                )
                response_body = json.loads(response["body"].read())
                if self.model_id.startswith("amazon.nova"):
                    embedding = response_body["embeddings"][0]["embedding"]
                elif "embedding" in response_body:
                    embedding = response_body["embedding"]
                elif "embeddings" in response_body:
                    embedding = response_body["embeddings"][0]
                else:
                    raise ValueError(f"Unexpected response format from {self.model_id}")
                all_embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error encoding text with Bedrock: {e}")
                raise
        return np.array(all_embeddings, dtype=np.float32)

    def get_model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            provider="bedrock",
            model_id=self.model_id,
            dimensions=self._dimension,
            max_tokens=self.config.get("max_tokens"),
            supports_normalization=self.config.get("supports_normalization", True),
            supports_variable_dimensions=self.config.get("supports_variable_dimensions", False),
            available_dimensions=self.config.get("available_dimensions"),
        )

    def get_embedding_dimension(self) -> int:
        return self._dimension


def create_embedding_provider(
    model_spec: str,
    device: Optional[str] = None,
    region_name: Optional[str] = None,
    dimensions: Optional[int] = None,
    **kwargs,
) -> EmbeddingProvider:
    if model_spec.startswith("bedrock:"):
        model_id = model_spec[8:]
        return BedrockEmbeddingProvider(model_id=model_id, region_name=region_name, dimensions=dimensions, **kwargs)
    elif model_spec.startswith("local:"):
        model_name = model_spec[6:]
        return LocalEmbeddingProvider(model_name=model_name, device=device, **kwargs)
    else:
        return LocalEmbeddingProvider(model_name=model_spec, device=device, **kwargs)


