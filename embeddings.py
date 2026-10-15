import numpy as np
from typing import List, Optional
import torch
import psutil

class LLMEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with adaptive memory configuration"""
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.available_memory_mb = 0
        
        if self.device == "cuda":
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.available_memory_mb = int(total_mem_gb * 1024 * 0.8)  # Use 80% of GPU memory
        
        print(f"Loading model on {self.device}...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def _calculate_optimal_batch_size(self, n_texts: int, target_memory_mb: float = 4096) -> int:
        """Calculate optimal batch size based on available GPU memory"""
        if not torch.cuda.is_available():
            return min(16, n_texts)  # CPU fallback
        
        # Estimate embedding size overhead (~50MB per batch for MiniLM)
        est_per_batch_mb = 50 + (n_texts // max(1, n_texts // 64)) * 10
        
        # Calculate safe batch size with memory headroom
        target_batch_gb = (target_memory_mb / 1024) / self.available_memory_mb if self.available_memory_mb > 0 else 1.0
        optimal_batch = int(n_texts * min(1.0, target_batch_gb))
        
        return max(1, min(optimal_batch, n_texts))
    
 1 | import numpy as np
 2 | from typing import List, Optional
 3 | import torch
 4 | import psutil
 5 | from datetime import datetime
 6 | import time
 7 | 
 8 | try:
 9 |     import aiohttp
10 |    from asyncio import Semaphore
11 | except ImportError:
12 |     aiohttp, Semaphore = None, None
13 | 
14 | class LLMEmbeddings:
15 |     def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
16 |         """Initialize with adaptive memory configuration"""
17 |         self.model = None
18 |         self.device = "cuda" if torch.cuda.is_available() else "cpu"
19 |         self.available_memory_mb = 0
20 |         self.metrics_collector = None
21 |         
22 |         if self.device == "cuda":
23 |             total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
24 |             self.available_memory_mb = int(total_mem_gb * 1024 * 0.8)
25 |             
26 |             # GPU Health Check - detect thermal throttling and power limits
27 |             gpu_stats = torch.cuda.get_device_properties(0).name
28 |             try:
29 |                 import pynvml
30 |                 pynvml.nvmlInit()
31 |                 handle = pynvml.nvmlDeviceGetHandleByIndex(0)
32 |                 temp_info = pynvml.nvmlDeviceGetTemperature(handle, 0)
33 |                 power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
34 |                 self.gpu_temp = temp_info
35 |                 self.gpu_power_watts = power_usage
36 |                 self.is_throttling = temp_info > 80 or power_usage < 20 and total_mem_gb > 8
37 |                 print(f"[GPU HEALTH] {gpu_stats}: Temp={temp_info}°C, Power={power_usage:.1f}W")
38 |             except Exception as e:
39 |                 self.gpu_temp = None
40 |                 self.gpu_power_watts = None
41 |                 self.is_throttling = False
42 |                 print(f"[GPU HEALTH] Monitoring disabled: {e}")
43 |         else:
44 |             self.gpu_temp = None
45 |             self.gpu_power_watts = None
46 |             self.is_throttling = False
47 |         
48 |         print(f"Loading model on {self.device}...")
49 |         from sentence_transformers import SentenceTransformer
50 |         self.model = SentenceTransformer(model_name, device=self.device)
51 |     
52 |     def _calculate_optimal_batch_size(self, n_texts: int, target_memory_mb: float = 4096) -> int:
53 |         """Calculate optimal batch size based on available GPU memory"""
54 |         if not torch.cuda.is_available():
55 |             return min(16, n_texts)
56 |         
57 |         est_per_batch_mb = 50 + (n_texts // max(1, n_texts // 64)) * 10
58 |         target_batch_gb = (target_memory_mb / 1024) / self.available_memory_mb if self.available_memory_mb > 0 else 1.0
59 |         optimal_batch = int(n_texts * min(1.0, target_batch_gb))
60 |         
61 |         # Adjust for thermal throttling
62 |         if hasattr(self, 'is_throttling') and self.is_throttling:
63 |             optimal_batch = max(1, optimal_batch // 2)
64 |         
65 |         return max(1, min(optimal_batch, n_texts))
66 |     
67 |     def _generate_batch(self, texts: List[str]) -> torch.Tensor:
68 |         """Internal method for single-batch inference with metrics tracking"""
69 |         start_time = time.time()
70 |         
71 |         with torch.no_grad():
72 |             encoded = self.model.encode(
73 |                 texts, 
74 |                 batch_size=len(texts),
75 |                 show_progress_bar=False,
76 |                 normalize_embeddings=True
77 |             )
78 |             
79 |         duration_ms = (time.time() - start_time) * 1000
        
    def generate_embeddings(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Generate embeddings with adaptive batching"""
        n_texts = len(texts)
        
        # Use provided batch_size or calculate optimal based on memory
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(n_texts)
        
        print(f"Generating {n_texts} embeddings (batch_size={batch_size}, device={self.device})")
        
        all_embeddings = []
        for i in range(0, n_texts, batch_size):
            chunk = texts[i:i + batch_size]
            embeddings = self._generate_batch(chunk)
            all_embeddings.extend(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings).astype(np.float32)
    
    def _generate_batch(self, texts: List[str]) -> torch.Tensor:
        """Internal method for single-batch inference"""
        with torch.no_grad():
            encoded = self.model.encode(
                texts, 
                batch_size=len(texts),  # Disable internal batching to use our calculated size
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return torch.tensor(encoded).to(self.device)