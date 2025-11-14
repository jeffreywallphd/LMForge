"""
LMForge Service Health Monitor
Monitors PostgreSQL, Ollama service availability and RAG status
Read-only health checking operations
"""
import time
import logging
import psycopg2
import requests
from typing import Dict, Any
from django.conf import settings
from django.db import connection

logger = logging.getLogger(__name__)


class ServiceHealthMonitor:
    """Monitor service health for LMForge - Read-only operations"""
    
    def __init__(self):
        self.postgres_config = {
            'host': getattr(settings, 'DB_HOST', 'localhost'),
            'port': getattr(settings, 'DB_PORT', '5435'),
            'database': getattr(settings, 'DB_NAME', 'pdf_rag_db'),
            'user': getattr(settings, 'DB_USER', 'pdf_rag_user'),
            'password': getattr(settings, 'DB_PASSWORD', 'pdf_rag_password')
        }
        
        self.ollama_url = getattr(settings, 'OLLAMA_URL', 'http://localhost:11434')
        
        # Status cache
        self.status_cache = {
            'postgres': {'available': False, 'last_check': 0, 'message': ''},
            'ollama': {'available': False, 'last_check': 0, 'message': ''},
            'rag_initialized': {'status': False, 'last_check': 0, 'message': ''}
        }
        self.cache_duration = 30  # seconds
    
    def check_postgres_health(self) -> Dict[str, Any]:
        """Check PostgreSQL service health"""
        try:
            # Try direct connection using psycopg2
            conn = psycopg2.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password'],
                connect_timeout=5
            )
            
            with conn.cursor() as cursor:
                # Check if pgvector extension exists
                cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                pgvector_exists = cursor.fetchone()[0]
                
                # Check if our core RAG tables exist (documents, chunks, sessions, conversations)
                cursor.execute("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name IN ('documents', 'chunks', 'sessions', 'conversations')
                """)
                tables_count = cursor.fetchone()[0]
                
            conn.close()
            
            return {
                'available': True,
                'pgvector_enabled': pgvector_exists,
                'tables_initialized': tables_count >= 4,  # Need at least 4 core tables
                'tables_count': tables_count,
                'message': f'PostgreSQL ready (pgvector: {pgvector_exists}, tables: {tables_count}/4)'
            }
            
        except Exception as e:
            return {
                'available': False,
                'pgvector_enabled': False,
                'tables_initialized': False,
                'message': f'PostgreSQL unavailable: {str(e)}'
            }
    
    def check_ollama_health(self) -> Dict[str, Any]:
        """Check Ollama service health with GPU/CPU detection"""
        try:
            # Import here to avoid circular imports
            from lmforge_core.services.clients.ollama_client import ollama_client
            
            # Check if Ollama is responding
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                # Check for required models
                embedding_model = getattr(settings, 'OLLAMA_EMBEDDING_MODEL', 'all-minilm:33m')
                chat_model = getattr(settings, 'OLLAMA_CHAT_MODEL', 'qwen2.5:0.5b-instruct')
                
                has_embedding = any(embedding_model in name for name in model_names)
                has_chat = any(chat_model in name for name in model_names)
                
                # Get GPU status
                gpu_status = ollama_client.get_gpu_status()
                mode = gpu_status.get('current_mode', 'CPU')
                
                return {
                    'available': True,
                    'models_loaded': len(models),
                    'embedding_ready': has_embedding,
                    'chat_ready': has_chat,
                    'gpu_available': gpu_status.get('gpu_available', False),
                    'mode': mode,
                    'recommended_parallel': gpu_status.get('recommended_parallel', 2),
                    'message': f'Ollama ready in {mode} mode ({len(models)} models: embedding={has_embedding}, chat={has_chat})'
                }
            else:
                return {
                    'available': False,
                    'models_loaded': 0,
                    'embedding_ready': False,
                    'chat_ready': False,
                    'gpu_available': False,
                    'mode': 'Unknown',
                    'message': f'Ollama responded with status {response.status_code}'
                }
                
        except Exception as e:
            return {
                'available': False,
                'models_loaded': 0,
                'embedding_ready': False,
                'chat_ready': False,
                'message': f'Ollama unavailable: {str(e)}'
            }
    
    def check_knowledge_base_status(self) -> Dict[str, Any]:
        """Check knowledge base status: JSON files and embeddings"""
        import os
        
        try:
            # Check if database tables exist
            postgres_status = self.get_postgres_status()
            if not postgres_status['available'] or not postgres_status['tables_initialized']:
                return {
                    'initialized': False,
                    'json_files_found': 0,
                    'json_files_stored': 0,
                    'total_chunks': 0,
                    'embeddings_generated': 0,
                    'message': 'Database not available or not initialized',
                    'files_status': []
                }
            
            # Find JSON files in media/JSON
            json_dir = os.path.join(settings.MEDIA_ROOT, 'JSON')
            json_files_found = []
            if os.path.exists(json_dir):
                json_files_found = [f for f in os.listdir(json_dir) if f.lower().endswith('.json')]
            
            # Check which files are stored in database
            with connection.cursor() as cursor:
                # Get stored documents
                cursor.execute("""
                    SELECT filename, total_chunks, processed 
                    FROM documents 
                    WHERE session_id IS NOT NULL
                """)
                stored_docs = cursor.fetchall()
                
                stored_files = {row[0]: {'chunks': row[1], 'processed': row[2]} 
                                for row in stored_docs}
                
                # Get total chunks with embeddings
                cursor.execute("""
                    SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL
                """)
                embeddings_count = cursor.fetchone()[0] or 0
                
                # Get total chunks
                cursor.execute("SELECT COUNT(*) FROM chunks")
                total_chunks = cursor.fetchone()[0] or 0
            
            # Build file status list
            files_status = []
            for json_file in json_files_found:
                if json_file in stored_files:
                    files_status.append({
                        'filename': json_file,
                        'stored': True,
                        'chunks': stored_files[json_file]['chunks'],
                        'processed': stored_files[json_file]['processed']
                    })
                else:
                    files_status.append({
                        'filename': json_file,
                        'stored': False,
                        'chunks': 0,
                        'processed': False
                    })
            
            json_files_stored = len(stored_files)
            initialized = json_files_stored > 0 and embeddings_count > 0
            
            return {
                'initialized': initialized,
                'json_files_found': len(json_files_found),
                'json_files_stored': json_files_stored,
                'total_chunks': total_chunks,
                'embeddings_generated': embeddings_count,
                'message': f'{json_files_stored}/{len(json_files_found)} JSON files processed, {embeddings_count} embeddings' if initialized else 'No data in knowledge base',
                'files_status': files_status
            }
            
        except Exception as e:
            logger.error(f"Error checking knowledge base status: {e}")
            return {
                'initialized': False,
                'json_files_found': 0,
                'json_files_stored': 0,
                'total_chunks': 0,
                'embeddings_generated': 0,
                'message': f'Error: {str(e)}',
                'files_status': []
            }
    
    def check_rag_initialization(self) -> Dict[str, Any]:
        """Check if RAG system is initialized with data (uses knowledge base status)"""
        kb_status = self.check_knowledge_base_status()
        return {
            'initialized': kb_status['initialized'],
            'documents_count': kb_status['json_files_stored'],
            'chunks_count': kb_status['total_chunks'],
            'message': kb_status['message']
        }
    
    def get_postgres_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get cached or fresh PostgreSQL status"""
        now = time.time()
        if force_refresh or (now - self.status_cache['postgres']['last_check']) > self.cache_duration:
            status = self.check_postgres_health()
            self.status_cache['postgres'] = {**status, 'last_check': now}
        
        return self.status_cache['postgres']
    
    def get_ollama_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get cached or fresh Ollama status"""
        now = time.time()
        if force_refresh or (now - self.status_cache['ollama']['last_check']) > self.cache_duration:
            status = self.check_ollama_health()
            self.status_cache['ollama'] = {**status, 'last_check': now}
        
        return self.status_cache['ollama']
    
    def get_rag_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get cached or fresh RAG initialization status"""
        now = time.time()
        if force_refresh or (now - self.status_cache['rag_initialized']['last_check']) > self.cache_duration:
            status = self.check_rag_initialization()
            self.status_cache['rag_initialized'] = {**status, 'last_check': now}
        
        return self.status_cache['rag_initialized']
    
    def get_system_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get complete system status including knowledge base"""
        postgres_status = self.get_postgres_status(force_refresh)
        ollama_status = self.get_ollama_status(force_refresh)
        rag_status = self.get_rag_status(force_refresh)
        kb_status = self.check_knowledge_base_status()
        
        # Determine overall system health
        system_ready = (
            postgres_status['available'] and 
            postgres_status.get('tables_initialized', False) and
            ollama_status['available'] and
            ollama_status.get('embedding_ready', False) and
            ollama_status.get('chat_ready', False)
        )
        
        return {
            'system_ready': system_ready,
            'postgres': postgres_status,
            'ollama': ollama_status,
            'rag': rag_status,
            'knowledge_base': kb_status,
            'timestamp': time.time()
        }


# Global instance
service_health = ServiceHealthMonitor()
