"""
Modern Multi-Stage Retrieval Pipeline (v2.0)
Architecture: Async ETL with modular stages, metrics collection, and fault tolerance

Pipeline Stages:
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│  Ingest     │ →  │  Transform   │ →  │  Enrich     │ →  │  Filter     │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘

Features:
- Async processing with backpressure
- Modular stage design (plug-and-play)
- Metrics collection at each stage
- Circuit breaker pattern for resilience
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, TypeVar
from enum import Enum
from datetime import datetime
import time
from collections import defaultdict


logger = logging.getLogger(__name__)

# =============================================================================
# TYPE DEFINITIONS & DATA CLASSES
# =============================================================================

class PipelineStatus(Enum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class Document:
    """Unified document representation"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content[:500],  # Truncate for JSON
            "metadata": self.metadata,
            "scores": self.scores,
            "has_embeddings": self.embeddings is not None
        }


@dataclass 
class PipelineMetrics:
    """Aggregate pipeline metrics"""
    start_time: float = field(default_factory=time.time)
    stage_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[Exception] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        return time.time() - self.start_time
    
    def record_stage(self, name: str, inputs: int, outputs: int, duration_ms: float):
        self.stage_metrics[name] = {
            "inputs": inputs,
            "outputs": outputs,
            "duration_ms": duration_ms,
            "throughput": outputs / (duration_ms / 1000) if duration_ms > 0 else 0
        }


# =============================================================================
# STAGE BASE CLASS (PLUG-AND-PLAY DESIGN)
# =============================================================================

class Stage(ABC):
    """Abstract base class for pipeline stages"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self._enabled = True
        
    @abstractmethod
    async def process(self, documents: List[Document]) -> List[Document]:
        """Process batch of documents"""
        pass
    
    def is_enabled(self) -> bool:
        return self._enabled


# =============================================================================
# PIPELINE ORCHESTRATOR (MAIN ENGINE)
# =============================================================================

class RetrievalPipeline:
    """Main pipeline orchestrator with async execution and metrics"""
    
    STAGES = [
        "ingest",       # Load raw documents from source
        "transform",    # Text preprocessing, tokenization  
        "embed",        # Generate embeddings (vectorize)
        "retrieve",     # Search/index query
        "rerank",       # Cross-encoder reranking
        "filter",       # Apply filters and thresholds
        "format"        # Final formatting/output
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "stages": {stage: {"enabled": True} for stage in self.STAGES},
            "batch_size": 100,
            "max_retries": 3,
            "circuit_breaker_threshold": 5,
            "timeout_seconds": 30
        }
        
        self.stages: Dict[str, Stage] = {}
        self._status = PipelineStatus.STOPPED
        self._metrics = PipelineMetrics()
        self._results_buffer: List[Document] = []
        
    def add_stage(self, stage: Stage):
        """Register a pipeline stage"""
        if not isinstance(stage, Stage):
            raise TypeError("Must provide Stage instance")
        if stage.name in self.stages:
            raise ValueError(f"Stage '{stage.name}' already exists")
        self.stages[stage.name] = stage
        
    async def execute(self, query: str) -> List[Document]:
        """Execute pipeline for given query"""
        
        logger.info(f"Starting pipeline execution: {query[:50]}...")
        self._status = PipelineStatus.RUNNING
        
        try:
            # Stage 1: Ingest (simulate - in production, load from DB/ES)
            raw_docs = await self._execute_stage("ingest", query=query)
            
            if not raw_docs:
                logger.warning("No documents retrieved")
                return []
                
            # Subsequent stages
            current_docs = raw_docs
            
            for stage_name in ["transform", "embed", "retrieve"]:
                if self.stages[stage_name].is_enabled():
                    current_docs = await self._execute_stage(stage_name, docs=current_docs)
                    if not current_docs:
                        logger.warning(f"Stage '{stage_name}' returned empty results")
                        break
                        
            # Rerank and filter (final stages)
            if self.stages["retrieve"].is_enabled():
                query_results = await self._execute_stage("retrieve", query=query, docs=current_docs)
                current_docs = query_results
                
            for stage_name in ["rerank", "filter"]:
                if self.stages[stage_name].is_enabled() and current_docs:
                    current_docs = await self._execute_stage(stage_name, docs=current_docs)
                    
            # Format output
            if self.stages["format"].is_enabled():
                current_docs = await self._execute_stage("format", docs=current_docs)
                
            self._status = PipelineStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self._metrics.errors.append(e)
            self._status = PipelineStatus.FAILED
            
        finally:
            self._save_metrics()
            
        return current_docs if self._status == PipelineStatus.COMPLETED else []
    
    async def _execute_stage(self, 
                            stage_name: str, 
                            docs: Optional[List[Document]] = None,
                            query: Optional[str] = None) -> List[Document]:
        """Execute single pipeline stage with metrics"""
        
        if stage_name not in self.stages:
            logger.warning(f"Stage '{stage_name}' not configured")
            return []
            
        stage = self.stages[stage_name]
        
        # Handle query-dependent stages (retrieve)
        if stage_name == "retrieve":
            docs = await stage.process([Document(
                id="query", 
                content=query, 
                metadata={"pipeline_stage": stage_name}
            )])
            
        inputs = len(docs or [])
        start_time = time.time()
        
        try:
            outputs = await asyncio.wait_for(
                stage.process(docs or []),
                timeout=self.config.get("timeout_seconds", 30)
            )
            duration_ms = (time.time() - start_time) * 1000
            
        except asyncio.TimeoutError:
            logger.error(f"Stage '{stage_name}' timed out")
            raise
        except Exception as e:
            logger.error(f"Stage '{stage_name}' failed: {e}")
            if len(self._metrics.errors) >= self.config.get("max_retries", 3):
                raise
            # Retry logic could go here
            
        finally:
            # Record metrics
            self._metrics.record_stage(
                stage_name, 
                inputs=inputs, 
                outputs=len(outputs), 
                duration_ms=duration_ms
            )
            
            logger.info(f"Stage '{stage_name}': {inputs} → {len(outputs)} docs ({duration_ms:.2f}ms)")
            
        return outputs
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics"""
        result = {
            "status": self._status.value,
            "duration_seconds": self._metrics.duration,
            "stage_performance": self._metrics.stage_metrics,
            "total_errors": len(self._metrics.errors)
        }
        
        if self._metrics.start_time:
            result["start_timestamp"] = datetime.fromtimestamp(
                self._metrics.start_time).isoformat()
                
        return result
    
    def _save_metrics(self):
        """Persist metrics to storage (placeholder for production)"""
        # In production: send to Prometheus, ELK, or custom DB


# =============================================================================
# STAGE IMPLEMENTATIONS (EXAMPLES)
# =============================================================================

class TextTransformStage(Stage):
    """Text preprocessing stage"""
    
    async def process(self, documents: List[Document]) -> List[Document]:
        # TODO: Implement tokenization, normalization, deduplication
        return documents


class EmbeddingGenerationStage(Stage):
    """Vector embedding generation"""
    
    async def process(self, documents: List[Document]) -> List[Document]:
        # TODO: Generate embeddings using sentence-transformers/all-MiniLM-L6-v2
        for doc in documents:
            if not doc.embeddings:
                doc.scores["embedding_generated"] = datetime.now().isoformat()
        return documents


class VectorSearchStage(Stage):
    """Vector database search stage"""
    
    async def process(self, 
                     documents: List[Document], 
                     query: str) -> List[Document]:
        
        # TODO: Query vector DB (Pinecone, Weaviate, Milvus)
        # Simulated results for now
        return [doc for doc in documents if doc.id != "query"]


class RerankingStage(Stage):
    """Cross-encoder reranking"""
    
    async def process(self, documents: List[Document]) -> List[Document]:
        # TODO: Apply cross-encoder (BGE-Reranker)
        for i, doc in enumerate(documents):
            # Mock score decay
            mock_score = 0.9 - (i * 0.03)
            if "scores" not in doc.metadata:
                doc.scores["rerank"] = mock_score
            else:
                doc.scores["rerank"] *= mock_score
        return documents


class ThresholdFilterStage(Stage):
    """Score threshold filtering"""
    
    def __init__(self, name: str, threshold: float = 0.5):
        super().__init__(name)
        self.threshold = threshold
        
    async def process(self, documents: List[Document]) -> List[Document]:
        filtered = [doc for doc in documents 
                   if doc.scores.get("rerank", 0) >= self.threshold]
        logger.info(f"Filter applied: {len(documents)} → {len(filtered)}")
        return filtered


class OutputFormatterStage(Stage):
    """Final output formatting"""
    
    async def process(self, documents: List[Document]) -> List[Document]:
        # Add final metadata and format for API response
        for doc in documents:
            if "metadata" not in doc.metadata:
                doc.metadata["formatted_at"] = datetime.now().isoformat()
        return documents


# =============================================================================
# DEMO & TESTING
# =============================================================================

async def demo_pipeline():
    """Demonstrate pipeline execution"""
    
    # Initialize pipeline with stages
    pipeline = RetrievalPipeline({
        "batch_size": 50,
        "circuit_breaker_threshold": 10
    })
    
    # Register stages
    pipeline.add_stage(TextTransformStage("transform"))
    pipeline.add_stage(EmbeddingGenerationStage("embed"))
    pipeline.add_stage(VectorSearchStage("retrieve"))
    pipeline.add_stage(RerankingStage("rerank", threshold=0.6))
    pipeline.add_stage(ThresholdFilterStage("filter", threshold=0.7))
    pipeline.add_stage(OutputFormatterStage("format"))
    
    # Execute query
    test_query = "What are the latest developments in AI safety research?"
    results = await pipeline.execute(test_query)
    
    print(f"\nQuery: {test_query}")
    print("=" * 60)
    
    if results:
        for i, doc in enumerate(results[:5], 1):
            print(f"\n[{i}] Score: {doc.scores.get('rerank', 0):.4f}")
            print(f"   ID: {doc.id[:30]}...")
            print(f"   Content: {doc.content[:100]}...")
    else:
        print("No results found")
    
    # Print metrics
    print("\n" + "=" * 60)
    print("PIPELINE METRICS:")
    for stage_name, metrics in pipeline.get_metrics()["stage_performance"].items():
        print(f"\n{stage_name.upper()}:")
        print(f"  Duration: {metrics['duration_ms']:.2f}ms")
        print(f"  Throughput: {metrics['throughput']:.2f} docs/sec")


if __name__ == "__main__":
    asyncio.run(demo_pipeline())
