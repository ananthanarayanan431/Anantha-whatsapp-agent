
import os 
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional

from anantha.settings import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


@dataclass
class Memory:
    """Represent a memory entry in the vector Qdrant vector store."""

    text: str 
    metadata: dict
    score: Optional[float] = None

    @property
    def id(self)-> Optional[str]:
        return self.metadata.get("id")
    
    @property
    def timestamp(self)-> Optional[datetime]:
        ts = self.metadata.get("timestamp")
        return datetime.fromisoformat(ts) if ts else None
    

class VectorStore:
    """A class to handle vector storage operations using Qdrant."""

    REQUIRED_ENV_VARS = ["QDRANT_URL", "QDRANT_API_KEY"]
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    COLLECTION_NAME = "long_term_memory"
    SIMILARITY_THRESHOLD = 0.9  

    _model = None
    _client = None 

    @classmethod
    def _validate_env_vars(cls)-> None:
        missing_vars = [var for var in cls.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
    @classmethod
    def _initialize(cls) -> None:
        if cls._model is None or cls._client is None:
            cls._validate_env_vars()
            cls._model = SentenceTransformer(cls.EMBEDDING_MODEL)
            cls._client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

    @classmethod
    def _collection_exists(cls) -> bool:
        cls._initialize()
        collections = cls._client.get_collections().collections
        return any(col.name == cls.COLLECTION_NAME for col in collections)
    
    @classmethod
    def _create_collection(cls) -> None:
        cls._initialize()
        sample_embedding = cls._model.encode("sample text")
        cls._client.create_collection(
            collection_name=cls.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=len(sample_embedding),
                distance=Distance.COSINE,
            ),
        )

    @classmethod
    def find_similar_memory(cls, text:str) -> Optional[Memory]:
        """Find if a similar memory already exists in the vector store.
        
        Args:
            text (str): The text to search for similar memories.
        
        Returns:
            Optional[Memory]: The most similar memory found, or None if no similar memory is found.
        """

        results = cls.search_memories(text, k=1) 
        if results and results[0].score > cls.SIMILARITY_THRESHOLD:
            return results[0]
        
        return None


    @classmethod
    def store_memory(cls, text:str, metadata:dict) -> None:
        """Store a new memory in the vector store or update if similar exists.
        
        Args:
            text (str): The text of the memory.
            metadata (dict): Additional information about the memory (timestamp, type, etc.)
        """

        if not cls._collection_exists():
            cls._create_collection()

        similar_memory = cls.find_similar_memory(text)
        if similar_memory and similar_memory.id:
            metadata["id"] = similar_memory.id  # Keep same ID for update

        embedding = cls._model.encode(text)
        point = PointStruct(
            id=metadata.get("id", hash(text)),
            vector=embedding.tolist(),
            payload={
                "text": text,
                **metadata,
            },
        )

        cls._client.upsert(
            collection_name=cls.COLLECTION_NAME,
            points=[point],
        )

    @classmethod
    def search_memories(cls, query: str, k:int = 5) -> List[Memory]:
        """Search for similar memories in the vector store.

        Args:
            query: Text to search for
            k: Number of results to return

        Returns:
            List of Memory objects
        """
        if not cls._collection_exists():
            return []
        
        query_embedding = cls._modelencode(query)
        results = cls._client.search(
            collection_name=cls.COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=k,
        )

        return [
            Memory(
                text=hit.payload["text"],
                metadata={k: v for k, v in hit.payload.items() if k != "text"},
                score=hit.score,
            )
            for hit in results
        ]


@lru_cache
def get_vector_store() -> type[VectorStore]:
    """Returns the class, since classmethods are used directly."""
    
    return VectorStore