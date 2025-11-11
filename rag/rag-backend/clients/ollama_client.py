"""
Simplified Ollama Client for Backend
Unified client for Ollama container with embedding and chat capabilities
"""
import requests
import logging
import os
from typing import List, Dict, Any, Optional
import concurrent.futures

logger = logging.getLogger(__name__)

class OllamaClient:
    """Simplified client for unified Ollama service"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.embedding_model = "all-minilm:33m"
        self.chat_model = "qwen2.5:0.5b-instruct"
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
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
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text"""
        if not text.strip():
            return None
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings else None
    
    def generate_embeddings(self, texts: List[str], use_gpu: bool = True) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not texts or not self._check_service():
            return []
        
        # Filter empty texts
        valid_texts = [(i, text) for i, text in enumerate(texts) if text.strip()]
        if not valid_texts:
            return []
        
        embeddings = [None] * len(texts)
        
        def generate_single_embedding(item):
            index, text = item
            try:
                response = self.session.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.embedding_model,
                        "prompt": text[:1000],  # Limit text length
                        "options": {
                            "num_ctx": 1024,
                            "num_batch": 256,
                            "num_gpu": 0,  # Force CPU mode
                            "num_thread": 1
                        }
                    },
                    timeout=15  # Shorter timeout
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
    
    def generate_chat_completion(self, messages: List[Dict[str, str]], system_message: str = None,
                               temperature: float = 0.3, max_tokens: int = 64, top_p: float = 0.8,
                               stream: bool = False) -> Optional[str]:
        """Generate chat completion with configurable parameters"""
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
            
            # Balance between stability and capability based on max_tokens
            if max_tokens > 512:
                # Enhanced mode for longer responses
                options = {
                    "temperature": temperature,
                    "num_ctx": 1024,  # Larger context for comprehensive responses
                    "num_predict": max_tokens,
                    "num_batch": 32,
                    "num_gpu": 0,  # Still force CPU for stability
                    "num_thread": 2,  # Allow 2 threads for better performance
                    "use_mlock": False,
                    "use_mmap": False,
                    "low_vram": True,
                    "f16_kv": True,
                    "repeat_penalty": 1.1,
                    "top_k": 30,
                    "top_p": top_p
                }
                timeout = 60  # Longer timeout for enhanced responses
            else:
                # Conservative mode for quick responses
                options = {
                    "temperature": temperature,
                    "num_ctx": 256,
                    "num_predict": max_tokens,
                    "num_batch": 16,
                    "num_gpu": 0,  # Force CPU mode for stability
                    "num_thread": 1,  # Single thread
                    "use_mlock": False,
                    "use_mmap": False,
                    "low_vram": True,
                    "f16_kv": True,
                    "repeat_penalty": 1.1,
                    "top_k": 20,
                    "top_p": top_p
                }
                timeout = 30  # Standard timeout
            
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
        """Get unified service status"""
        available = self._check_service()
        return {
            "ollama": {
                "available": available,
                "url": self.base_url,
                "embedding_model": self.embedding_model,
                "chat_model": self.chat_model
            },
            "service_available": available
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

# Global instance
ollama_client = OllamaClient()