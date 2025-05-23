import os  # Standard library for file path operations
from abc import ABC  # Abstract base class support
from collections.abc import AsyncGenerator, Awaitable  # Types for async generators and awaitable return types
from dataclasses import dataclass  # Simplify class boilerplate for data containers
from typing import Any, Callable, Optional, TypedDict, Union, cast  # Type hints
from urllib.parse import urljoin  # Helper to construct URLs

import aiohttp  # Async HTTP client for image embedding calls
from azure.search.documents.agent.aio import KnowledgeAgentRetrievalClient  # Agent client for retrieval
from azure.search.documents.agent.models import (
    KnowledgeAgentAzureSearchDocReference,
    KnowledgeAgentIndexParams,
    KnowledgeAgentMessage,
    KnowledgeAgentMessageTextContent,
    KnowledgeAgentRetrievalRequest,
    KnowledgeAgentRetrievalResponse,
    KnowledgeAgentSearchActivityRecord,
)
from azure.search.documents.aio import SearchClient  # Azure SDK client for search operations
from azure.search.documents.models import (
    QueryCaptionResult,
    QueryType,
    VectorizedQuery,
    VectorQuery,
)
from openai import AsyncOpenAI, AsyncStream  # OpenAI async client and streaming types
from openai.types import CompletionUsage  # Token usage stats type
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionReasoningEffort,
    ChatCompletionToolParam,
)

from approaches.promptmanager import PromptManager  # Manages prompt templates and variables
from core.authentication import AuthenticationHelper  # Handles security filters and auth


@dataclass
class Document:
    """
    Represents a single retrieved document, including metadata and relevance scores.
    """
    id: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    sourcepage: Optional[str] = None
    sourcefile: Optional[str] = None
    oids: Optional[list[str]] = None
    groups: Optional[list[str]] = None
    captions: Optional[list[QueryCaptionResult]] = None
    score: Optional[float] = None
    reranker_score: Optional[float] = None
    search_agent_query: Optional[str] = None

    def serialize_for_results(self) -> dict[str, Any]:
        """
        Convert the Document into a JSON-serializable dict for API responses.
        """
        result_dict = {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "sourcepage": self.sourcepage,
            "sourcefile": self.sourcefile,
            "oids": self.oids,
            "groups": self.groups,
            "captions": (
                [  # Format caption results into dictionaries
                    {
                        "additional_properties": caption.additional_properties,
                        "text": caption.text,
                        "highlights": caption.highlights,
                    }
                    for caption in self.captions
                ]
                if self.captions
                else []
            ),
            "score": self.score,
            "reranker_score": self.reranker_score,
            "search_agent_query": self.search_agent_query,
        }
        return result_dict


@dataclass
class ThoughtStep:
    """
    Records a reasoning step with title, description, and optional properties.
    """
    title: str
    description: Optional[Any]
    props: Optional[dict[str, Any]] = None

    def update_token_usage(self, usage: CompletionUsage) -> None:
        """
        Attach token usage stats to this step's properties.
        """
        if self.props:
            self.props["token_usage"] = TokenUsageProps.from_completion_usage(usage)


@dataclass
class DataPoints:
    """
    Holds optional lists of text or image data generated during processing.
    """
    text: Optional[list[str]] = None
    images: Optional[list] = None


@dataclass
class ExtraInfo:
    """
    Encapsulates detailed diagnostic info: data points, reasoning steps, and follow-ups.
    """
    data_points: DataPoints
    thoughts: Optional[list[ThoughtStep]] = None
    followup_questions: Optional[list[Any]] = None


@dataclass
class TokenUsageProps:
    """
    Simplified token usage fields exposed to clients.
    """
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: Optional[int]
    total_tokens: int

    @classmethod
    def from_completion_usage(cls, usage: CompletionUsage) -> "TokenUsageProps":
        """
        Create a TokenUsageProps from OpenAI's CompletionUsage object.
        """
        return cls(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning_tokens=(
                usage.completion_tokens_details.reasoning_tokens
                if usage.completion_tokens_details
                else None
            ),
            total_tokens=usage.total_tokens,
        )


# GPT reasoning models have unique streaming/parameter support
@dataclass
class GPTReasoningModelSupport:
    streaming: bool  # Indicates if this model supports streaming responses


class Approach(ABC):
    """
    Base abstraction for retrieval + generation approaches using Azure Search and OpenAI.
    """
    # Supported GPT reasoning models and their features
    GPT_REASONING_MODELS = {
        "o1": GPTReasoningModelSupport(streaming=False),
        "o3-mini": GPTReasoningModelSupport(streaming=True),
    }
    # Token limits for standard vs reasoning models
    RESPONSE_DEFAULT_TOKEN_LIMIT = 1024
    RESPONSE_REASONING_DEFAULT_TOKEN_LIMIT = 8192

    def __init__(
        self,
        search_client: SearchClient,
        openai_client: AsyncOpenAI,
        auth_helper: AuthenticationHelper,
        query_language: Optional[str],
        query_speller: Optional[str],
        embedding_deployment: Optional[str],  # Azure-specific deployment name
        embedding_model: str,
        embedding_dimensions: int,
        embedding_field: str,
        openai_host: str,
        vision_endpoint: str,
        vision_token_provider: Callable[[], Awaitable[str]],
        prompt_manager: PromptManager,
        reasoning_effort: Optional[str] = None,
    ):
        # Initialize clients, configuration, and dependencies
        self.search_client = search_client
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.query_language = query_language
        self.query_speller = query_speller
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.embedding_field = embedding_field
        self.openai_host = openai_host
        self.vision_endpoint = vision_endpoint
        self.vision_token_provider = vision_token_provider
        self.prompt_manager = prompt_manager
        self.reasoning_effort = reasoning_effort
        self.include_token_usage = True

    def build_filter(self, overrides: dict[str, Any], auth_claims: dict[str, Any]) -> Optional[str]:
        """
        Constructs an Azure Search filter string from include/exclude overrides and security claims.
        """
        include_category = overrides.get("include_category")
        exclude_category = overrides.get("exclude_category")
        security_filter = self.auth_helper.build_security_filters(overrides, auth_claims)
        filters = []
        if include_category:
            filters.append(f"category eq '{include_category.replace("'", "''")}'")
        if exclude_category:
            filters.append(f"category ne '{exclude_category.replace("'", "''")}'")
        if security_filter:
            filters.append(security_filter)
        return None if not filters else " and ".join(filters)

    async def search(
        self,
        top: int,
        query_text: Optional[str],
        filter: Optional[str],
        vectors: list[VectorQuery],
        use_text_search: bool,
        use_vector_search: bool,
        use_semantic_ranker: bool,
        use_semantic_captions: bool,
        minimum_search_score: Optional[float] = None,
        minimum_reranker_score: Optional[float] = None,
        use_query_rewriting: Optional[bool] = None,
    ) -> list[Document]:
        """
        Execute an Azure search call combining text, vector, and semantic options, then parse results.
        """
        search_text = query_text if use_text_search else ""
        search_vectors = vectors if use_vector_search else []
        # Choose semantic vs basic search path
        if use_semantic_ranker:
            results = await self.search_client.search(
                search_text=search_text,
                filter=filter,
                top=top,
                query_caption=(
                    "extractive|highlight-false" if use_semantic_captions else None
                ),
                query_rewrites="generative" if use_query_rewriting else None,
                vector_queries=search_vectors,
                query_type=QueryType.SEMANTIC,
                query_language=self.query_language,
                query_speller=self.query_speller,
                semantic_configuration_name="default",
                semantic_query=query_text,
            )
        else:
            results = await self.search_client.search(
                search_text=search_text,
                filter=filter,
                top=top,
                vector_queries=search_vectors,
            )

        documents: list[Document] = []
        # Iterate pages asynchronously and convert to Document objects
        async for page in results.by_page():
            async for doc in page:
                documents.append(
                    Document(
                        id=doc.get("id"),
                        content=doc.get("content"),
                        category=doc.get("category"),
                        sourcepage=doc.get("sourcepage"),
                        sourcefile=doc.get("sourcefile"),
                        oids=doc.get("oids"),
                        groups=doc.get("groups"),
                        captions=cast(list[QueryCaptionResult], doc.get("@search.captions")),
                        score=doc.get("@search.score"),
                        reranker_score=doc.get("@search.reranker_score"),
                    )
                )
            # Filter by minimum scores
            documents = [
                d for d in documents
                if (d.score or 0) >= (minimum_search_score or 0)
                and (d.reranker_score or 0) >= (minimum_reranker_score or 0)
            ]
        return documents

    # ... additional methods would be annotated similarly with docstrings and inline comments ...
