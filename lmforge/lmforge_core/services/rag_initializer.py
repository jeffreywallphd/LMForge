"""
RAG System Initializer
Handles database table initialization and RAG storage setup
Write operations for system setup
"""
import os
import logging
import psycopg2
from typing import Dict, Any
from django.conf import settings
from django.core.management import call_command

logger = logging.getLogger(__name__)


class RAGInitializer:
    """Initialize RAG system components - Write operations"""
    
    def __init__(self):
        self.postgres_config = {
            'host': getattr(settings, 'DB_HOST', 'localhost'),
            'port': getattr(settings, 'DB_PORT', '5435'),
            'database': getattr(settings, 'DB_NAME', 'pdf_rag_db'),
            'user': getattr(settings, 'DB_USER', 'pdf_rag_user'),
            'password': getattr(settings, 'DB_PASSWORD', 'pdf_rag_password')
        }
    
    def initialize_database_tables(self) -> Dict[str, Any]:
        """Initialize database tables using init-db.sql if needed"""
        from .service_health import service_health
        
        try:
            postgres_status = service_health.get_postgres_status()
            if not postgres_status['available']:
                return {'success': False, 'message': 'PostgreSQL not available'}
            
            if postgres_status['tables_initialized']:
                return {'success': True, 'message': 'Database tables already initialized'}
            
            # Run init-db.sql with error handling for existing objects
            init_sql_path = os.path.join(settings.BASE_DIR, 'init-db.sql')
            if not os.path.exists(init_sql_path):
                return {'success': False, 'message': 'init-db.sql not found'}
            
            conn = psycopg2.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config['port'],
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password']
            )
            conn.autocommit = True  # Use autocommit to handle errors gracefully
            
            with open(init_sql_path, 'r') as f:
                sql_content = f.read()
            
            # Execute SQL with proper error handling for existing objects
            with conn.cursor() as cursor:
                try:
                    cursor.execute(sql_content)
                except psycopg2.errors.DuplicateObject as e:
                    # Triggers, functions, etc. already exist - this is fine
                    logger.info(f"Some objects already exist (expected): {e}")
                except psycopg2.errors.DuplicateTable as e:
                    # Tables already exist - this is fine
                    logger.info(f"Tables already exist (expected): {e}")
                except Exception as e:
                    error_msg = str(e)
                    # Check if error is about existing objects
                    if any(keyword in error_msg.lower() for keyword in ['already exists', 'duplicate']):
                        logger.info(f"Skipping existing database objects: {error_msg}")
                    else:
                        conn.close()
                        return {'success': False, 'message': f'Database initialization error: {error_msg}'}
            
            conn.close()
            
            # Clear cache to force recheck
            service_health.status_cache['postgres']['last_check'] = 0
            
            return {'success': True, 'message': 'Database tables initialized successfully'}
            
        except Exception as e:
            return {'success': False, 'message': f'Database initialization failed: {str(e)}'}
    
    def initialize_rag_storage(self) -> Dict[str, Any]:
        """Initialize RAG storage from JSON files"""
        from .service_health import service_health
        
        try:
            # Check if services are ready
            if not service_health.get_ollama_status()['available']:
                return {'success': False, 'message': 'Ollama service not available'}
            
            postgres_status = service_health.get_postgres_status()
            if not postgres_status['available'] or not postgres_status['tables_initialized']:
                return {'success': False, 'message': 'PostgreSQL not ready'}
            
            # Run the management command
            from io import StringIO
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                call_command('init_rag_storage', '--force')
                output = sys.stdout.getvalue()
                
                # Clear cache to force recheck
                service_health.status_cache['rag_initialized']['last_check'] = 0
                
                return {'success': True, 'message': 'RAG storage initialized successfully', 'output': output}
                
            finally:
                sys.stdout = old_stdout
            
        except Exception as e:
            return {'success': False, 'message': f'RAG initialization failed: {str(e)}'}


# Global instance
rag_initializer = RAGInitializer()
