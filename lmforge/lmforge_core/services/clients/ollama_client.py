"""
Django-adapted Ollama Client for LMForge
Unified client for Ollama container with embedding and chat capabilities
"""
import requests
import requests.adapters
import logging
import os
from typing import List, Dict, Any, Optional
import concurrent.futures
from django.conf import settings

logger = logging.getLogger(__name__)

class OllamaClient:
    """Simplified client for unified Ollama service with GPU/CPU detection"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or getattr(settings, 'OLLAMA_URL', os.getenv("OLLAMA_URL", "http://localhost:11434"))
        self.embedding_model = getattr(settings, 'OLLAMA_EMBEDDING_MODEL', "all-minilm:33m")
        self.chat_model = getattr(settings, 'OLLAMA_CHAT_MODEL', "qwen2.5:0.5b-instruct")
        
        # Session for connection pooling with retry adapter
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        # Configure session with longer keep-alive and retry settings
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=3,
            pool_block=False
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # GPU detection and configuration
        self._gpu_available = None
        self._system_info = None
        
        logger.info(f"Ollama client initialized with URL: {self.base_url}")
        logger.info(f"Using embedding model: {self.embedding_model}")
        logger.info(f"Using chat model: {self.chat_model}")
    
    def _check_service(self) -> bool:
        """Check if unified Ollama service is available"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=3)
            return response.status_code == 200
        except Exception:
            return False
    
    def _detect_gpu_capability(self) -> Dict[str, Any]:
        """Detect GPU availability and system capabilities"""
        if self._system_info is not None:
            return self._system_info
        
        system_info = {
            "gpu_available": False,
            "gpu_memory": 0,
            "cpu_threads": 2,
            "recommended_parallel": 2,
            "recommended_gpu_layers": 0
        }
        
        try:
            # Try to get system information from Ollama
            response = self.session.get(f"{self.base_url}/api/ps", timeout=5)
            if response.status_code == 200:
                ps_data = response.json()
                
                # Look for GPU information in running models
                models = ps_data.get("models", [])
                for model in models:
                    if "gpu" in model.get("details", {}).get("format", "").lower():
                        system_info["gpu_available"] = True
                        break
            
            # Alternative: Check via model generation with GPU test
            if not system_info["gpu_available"]:
                system_info = self._test_gpu_capability()
            
        except Exception as e:
            logger.debug(f"GPU detection failed: {e}, falling back to CPU mode")
        
        # Cache the result
        self._system_info = system_info
        self._gpu_available = system_info["gpu_available"]
        
        logger.info(f"System capabilities detected: GPU={system_info['gpu_available']}, "
                   f"Parallel={system_info['recommended_parallel']}")
        
        return system_info
    
    def _test_gpu_capability(self) -> Dict[str, Any]:
        """Test GPU capability by attempting a small GPU-enabled inference"""
        system_info = {
            "gpu_available": False,
            "gpu_memory": 0,
            "cpu_threads": 2,
            "recommended_parallel": 2,
            "recommended_gpu_layers": 0
        }
        
        try:
            # Test with a simple GPU request
            test_response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.chat_model,
                    "prompt": "Test",
                    "stream": False,
                    "options": {
                        "num_predict": 1,
                        "num_gpu": 32,  # Try to use GPU
                        "num_ctx": 128
                    }
                },
                timeout=10
            )
            
            if test_response.status_code == 200:
                # If GPU request succeeds, we likely have GPU
                system_info.update({
                    "gpu_available": True,
                    "recommended_parallel": 6,
                    "recommended_gpu_layers": 32
                })
                logger.info("GPU capability confirmed through test inference")
            else:
                logger.info("GPU test failed, using CPU mode")
                
        except Exception as e:
            logger.debug(f"GPU test failed: {e}")
        
        return system_info
    
    def _get_optimal_options(self, use_gpu: Optional[bool] = None, task_type: str = "chat") -> Dict[str, Any]:
        """Get optimal options based on detected capabilities"""
        system_info = self._detect_gpu_capability()
        
        # Auto-detect if not specified
        if use_gpu is None:
            use_gpu = system_info["gpu_available"]
        
        # Force CPU if GPU not available
        if use_gpu and not system_info["gpu_available"]:
            use_gpu = False
            logger.info("GPU requested but not available, falling back to CPU")
        
        if use_gpu and system_info["gpu_available"]:
            # GPU optimized settings
            if task_type == "embedding":
                return {
                    "num_ctx": 1024,
                    "num_batch": 512,
                    "num_gpu": system_info["recommended_gpu_layers"],
                    "num_thread": 1
                }
            else:  # chat
                return {
                    "num_ctx": 2048,
                    "num_batch": 64,
                    "num_gpu": system_info["recommended_gpu_layers"],
                    "num_thread": 1,
                    "use_mlock": True,
                    "use_mmap": True,
                    "f16_kv": True
                }
        else:
            # CPU optimized settings
            if task_type == "embedding":
                return {
                    "num_ctx": 1024,
                    "num_batch": 256,
                    "num_gpu": 0,
                    "num_thread": system_info["cpu_threads"]
                }
            else:  # chat
                return {
                    "num_ctx": 1024,
                    "num_batch": 32,
                    "num_gpu": 0,
                    "num_thread": system_info["cpu_threads"],
                    "use_mlock": False,
                    "use_mmap": False,
                    "low_vram": True,
                    "f16_kv": True
                }
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text"""
        if not text.strip():
            return None
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings else None
    
    def generate_embeddings(self, texts: List[str], use_gpu: Optional[bool] = None) -> List[List[float]]:
        """Generate embeddings for multiple texts with GPU/CPU auto-detection"""
        if not texts or not self._check_service():
            return []
        
        # Filter empty texts
        valid_texts = [(i, text) for i, text in enumerate(texts) if text.strip()]
        if not valid_texts:
            return []
        
        # Get optimal options based on GPU availability
        optimal_options = self._get_optimal_options(use_gpu, "embedding")
        
        embeddings = [None] * len(texts)
        
        def generate_single_embedding(item):
            index, text = item
            try:
                response = self.session.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text[:1000],  # Limit text length
                        "options": optimal_options
                    },
                    timeout=30  # Extended timeout for embeddings
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return index, result.get("embedding", [])
                else:
                    logger.error(f"Embedding API error: {response.status_code}")
                    return index, []
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                return index, []
        
        try:
            # Parallel processing
            max_workers = min(4, len(valid_texts))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(generate_single_embedding, valid_texts))
            
            # Place results in correct positions
            for index, embedding in results:
                embeddings[index] = embedding
            
            # Filter out None values
            valid_embeddings = [emb for emb in embeddings if emb]
            logger.info(f"Generated {len(valid_embeddings)}/{len(texts)} embeddings")
            return valid_embeddings
            
        except Exception as e:
            logger.error(f"Error in embedding generation: {e}")
            return []
    
    def generate_chat_completion(self, messages: List[Dict[str, str]], system_message: Optional[str] = None,
                               temperature: float = 0.3, max_tokens: int = 64, top_p: float = 0.8,
                               stream: bool = False, use_gpu: Optional[bool] = None) -> Optional[str]:
        """Generate chat completion with GPU/CPU auto-detection and configurable parameters"""
        if not self._check_service():
            logger.error("Ollama service not available")
            return None
        
        try:
            # Prepare prompt
            prompt_parts = []
            if system_message:
                prompt_parts.append(f"System: {system_message}")
            
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                prompt_parts.append(f"{role.title()}: {content}")
            
            prompt = "\n".join(prompt_parts) + "\nAssistant:"
            
            # Get optimal options based on GPU availability and task complexity
            base_options = self._get_optimal_options(use_gpu, "chat")
            
            # Adjust based on response length requirements
            if max_tokens > 512:
                # Enhanced mode for longer responses
                options = {
                    **base_options,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "repeat_penalty": 1.1,
                    "top_k": 30,
                    "top_p": top_p
                }
                timeout = 120  # Extended timeout for enhanced responses (2 minutes)
            else:
                # Conservative mode for quick responses
                options = {
                    **base_options,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "repeat_penalty": 1.1,
                    "top_k": 20,
                    "top_p": top_p
                }
                # Adjust context and batch size for quick responses
                options["num_ctx"] = min(options.get("num_ctx", 1024), 512)
                options["num_batch"] = min(options.get("num_batch", 32), 16)
                timeout = 90  # Extended standard timeout (1.5 minutes)
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.chat_model,
                    "prompt": prompt,
                    "stream": stream,
                    "options": options
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()
                logger.info(f"Generated response: {len(response_text)} chars with max_tokens={max_tokens}")
                return response_text
            else:
                logger.error(f"Chat API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating chat completion: {e}")
            return None
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get unified service status with GPU/CPU information"""
        available = self._check_service()
        system_info = self._detect_gpu_capability() if available else {}
        
        return {
            "ollama": {
                "available": available,
                "url": self.base_url,
                "embedding_model": self.embedding_model,
                "chat_model": self.chat_model,
                "gpu_available": system_info.get("gpu_available", False),
                "mode": "GPU" if system_info.get("gpu_available", False) else "CPU",
                "recommended_parallel": system_info.get("recommended_parallel", 2)
            },
            "service_available": available,
            "gpu_enabled": system_info.get("gpu_available", False)
        }
    
    def check_model_availability(self) -> Dict[str, Any]:
        """Check if models are available"""
        models = []
        
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                model_data = response.json().get("models", [])
                models = [model.get("name", "") for model in model_data]
        except Exception as e:
            logger.error(f"Error checking models: {e}")
        
        embedding_available = any(self.embedding_model in name for name in models)
        chat_available = any(self.chat_model in name for name in models)
        
        return {
            "models": models,
            "embedding_model": self.embedding_model,
            "chat_model": self.chat_model,
            "embedding_available": embedding_available,
            "chat_available": chat_available,
            "service_available": bool(models)
        }
    
    def health_check(self) -> bool:
        """Quick health check"""
        return self._check_service()
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """Get detailed GPU/CPU status and recommendations"""
        if not self._check_service():
            return {
                "service_available": False,
                "error": "Ollama service not available"
            }
        
        system_info = self._detect_gpu_capability()
        
        return {
            "service_available": True,
            "gpu_available": system_info["gpu_available"],
            "current_mode": "GPU" if system_info["gpu_available"] else "CPU",
            "gpu_memory": system_info.get("gpu_memory", 0),
            "cpu_threads": system_info["cpu_threads"],
            "recommended_parallel": system_info["recommended_parallel"],
            "recommended_gpu_layers": system_info.get("recommended_gpu_layers", 0),
            "performance_tips": self._get_performance_tips(system_info)
        }
    
    def _get_performance_tips(self, system_info: Dict[str, Any]) -> List[str]:
        """Get performance optimization tips based on system capabilities"""
        tips = []
        
        if system_info["gpu_available"]:
            tips.extend([
                "GPU acceleration is available and enabled",
                f"Recommended parallel requests: {system_info['recommended_parallel']}",
                "Using GPU layers for faster inference"
            ])
        else:
            tips.extend([
                "Running in CPU mode - consider GPU for better performance",
                f"Optimized for {system_info['cpu_threads']} CPU threads",
                f"Recommended parallel requests: {system_info['recommended_parallel']}",
                "Using memory-optimized settings for CPU inference"
            ])
        
        return tips

# Global instance
ollama_client = OllamaClient()