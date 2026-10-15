import numpy as np
from datetime import datetime
import psutil


class EmbeddingsMetricsCollector:
    """Collects and analyzes performance metrics for embedding generation"""
    
    def __init__(self):
        self.metrics = []
        self._lock = False  # Thread safety placeholder
    
    def track_embedding_generation(
        self, 
        n_texts: int, 
        batch_size: int, 
        duration_ms: float,
        device: str = "cpu"
    ):
        """Track performance metrics for embeddings generation"""
        throughput = (n_texts / duration_ms) * 1000 if duration_ms > 0 else 0
        memory_usage = psutil.virtual_memory().percent
        
        metric = {
            "timestamp": datetime.now().isoformat(),
            "texts_count": n_texts,
            "batch_size": batch_size,
            "duration_ms": round(duration_ms, 2),
            "throughput": round(throughput, 2),
            "memory_percent": memory_usage,
            "device": device
        }
        
        self.metrics.append(metric)
        print(f"[METRICS] {n_texts} texts @ batch={batch_size}: "
              f"{duration_ms:.2f}ms | {throughput:.1f} texts/sec")
    
    def get_optimal_batch_recommendation(self, target_throughput: float = 500.0) -> int:
        """Analyze history and recommend optimal batch size"""
        if len(self.metrics) < 3:
            return 32
        
        recent = self.metrics[-10:]
        
        # Find batches with good throughput
        high_throughput_batches = [m for m in recent if m["throughput"] >= target_throughput]
        
        if not high_throughput_batches:
            # Return most common batch size from history
            batch_counts = {}
            for m in self.metrics[-10:]:
                bs = m["batch_size"]
                batch_counts[bs] = batch_counts.get(bs, 0) + 1
            
            return max(batch_counts, key=batch_counts.get) if batch_counts else 32
        
        # Return most common high-throughput batch size
        batch_counts = {}
        for m in high_throughput_batches:
            bs = m["batch_size"]
            batch_counts[bs] = batch_counts.get(bs, 0) + 1
        
        return max(batch_counts, key=batch_counts.get) if batch_counts else 32
    
    def get_performance_summary(self) -> dict:
        """Get summary statistics from collected metrics"""
        if not self.metrics:
            return {"error": "No metrics available"}
        
        durations = [m["duration_ms"] for m in self.metrics]
        throughputs = [m["throughput"] for m in self.metrics]
        batch_sizes = [m["batch_size"] for m in self.metrics]
        
        return {
            "total_metrics": len(self.metrics),
            "avg_duration_ms": np.mean(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "avg_throughput": np.mean(throughputs),
            "recommended_batch_size": self.get_optimal_batch_recommendation(),
            "devices_used": list(set(m["device"] for m in self.metrics)),
            "latest_metric": self.metrics[-1] if self.metrics else None
        }


# Global instance for easy access
metrics_collector = EmbeddingsMetricsCollector()
