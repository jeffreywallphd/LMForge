"""
Django-adapted Chat Service for RAG-based LLM Interactions
Handles chat sessions, context retrieval, and LLM communication
"""
import logging
import uuid
import time
from typing import List, Dict, Any, Optional
import json
from django.db import connection
from django.conf import settings

from .clients.ollama_client import ollama_client
from .embedding_client import embedding_client

logger = logging.getLogger(__name__)

class ChatService:
    
    def __init__(self):
        self.chat_model = getattr(settings, 'OLLAMA_CHAT_MODEL', "qwen2.5:0.5b-instruct")
        self.default_model = self.chat_model  # Alias for API compatibility
        self.max_context_chunks = 8  # Increased for more aggressive search
        self.max_chat_history = 10  # Increased for better context
        self.similarity_threshold = 0.6  # Lower threshold for more results
        
        logger.info("ChatService initialized - optimized sync operations for fast performance")
    
    def create_chat_session_sync(self, user_id: Optional[str] = None, session_name: Optional[str] = None) -> str:
        """
        Create a new chat session in PostgreSQL using sync connection
        Args:
            user_id: Optional user identifier
            session_name: Optional session name
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO sessions (id, status, created_at, updated_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, [session_id, 'active'])
            
            logger.info(f"Created chat session in PostgreSQL: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating chat session: {e}")
            return session_id  # Return anyway for graceful degradation

    def create_chat_session(self, user_id: Optional[str] = None, session_name: Optional[str] = None) -> str:
        """
        Create a new chat session in PostgreSQL - optimized sync version
        Args:
            user_id: Optional user identifier
            session_name: Optional session name
        Returns:
            Session ID
        """
        return self.create_chat_session_sync(user_id, session_name)
    
    def get_chat_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get chat session data from PostgreSQL - optimized sync version"""
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT id, status, created_at, updated_at
                    FROM sessions
                    WHERE id = %s
                """, [session_id])
                
                row = cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'status': row[1], 
                        'created_at': row[2],
                        'updated_at': row[3]
                    }
                return None
            
        except Exception as e:
            logger.error(f"Error getting chat session: {e}")
            return None
    
    def get_chat_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get chat history for a session from PostgreSQL - optimized sync version
        Args:
            session_id: Chat session ID
            limit: Maximum number of messages to retrieve
        Returns:
            List of chat messages
        """
        return self.get_chat_history_sync(session_id, limit)
    
    def store_chat_message(self, session_id: str, role: str, content: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Store a chat message in PostgreSQL - optimized sync version
        Args:
            session_id: Chat session ID
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Additional metadata
        Returns:
            Message ID (integer)
        """
        return self.store_chat_message_sync(session_id, role, content, metadata)
    
    def retrieve_relevant_context(self, query: str, session_id: Optional[str] = None, 
                                 top_k: Optional[int] = None,
                                 storage_backend: str = 'pgvector') -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from the knowledge base using aggressive RAG search"""
        top_k = top_k or self.max_context_chunks
        
        try:
            # Get recent chat history for enhanced query context
            chat_history = self.get_chat_history(session_id, limit=self.max_chat_history) if session_id else []
            
            # Create enhanced query by combining current message with recent context
            enhanced_query_parts = [query]
            if chat_history:
                # Add last few user messages for context
                recent_user_messages = [msg['content'] for msg in chat_history[-3:] if msg['message_type'] == 'user']
                if recent_user_messages:
                    enhanced_query_parts.extend(recent_user_messages)
            
            enhanced_query = " ".join(enhanced_query_parts)
            logger.info(f"Enhanced query with chat history: {len(enhanced_query)} chars")
            
            # Generate embedding for the enhanced query
            query_embedding = ollama_client.generate_embedding(enhanced_query)
            if not query_embedding:
                # Fallback to original query
                query_embedding = ollama_client.generate_embedding(query)
                if not query_embedding:
                    return []
            
            # Aggressive search: multiple searches with different strategies
            all_chunks = []
            
            # Import rag_database for search operations
            from .rag_database import rag_database
            
            # 1. Primary search with enhanced query - using sync method
            primary_chunks = rag_database.search_similar_chunks_sync(
                query_embedding, 
                top_k=top_k,
                threshold=self.similarity_threshold,
                storage_backend=storage_backend
            )
            all_chunks.extend(primary_chunks)
            
            # 2. Secondary search with original query (if different)
            if enhanced_query != query:
                original_embedding = ollama_client.generate_embedding(query)
                if original_embedding:
                    secondary_chunks = rag_database.search_similar_chunks_sync(
                        original_embedding,
                        top_k=top_k//2,
                        threshold=self.similarity_threshold - 0.1,  # Lower threshold
                        storage_backend=storage_backend
                    )
                    all_chunks.extend(secondary_chunks)
            
            # Remove duplicates and rank by similarity
            seen_chunks = set()
            unique_chunks = []
            for chunk in all_chunks:
                chunk_id = chunk.get("chunk_id")
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    unique_chunks.append(chunk)
            
            # Sort by similarity score and take top results
            unique_chunks.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            context_chunks = unique_chunks[:top_k]
            
            # Enhance chunks with relevance scores and metadata
            enhanced_context = []
            for chunk in context_chunks:
                context_chunk = {
                    "chunk_id": chunk.get("chunk_id"),
                    "content": chunk.get("content", ""),
                    "document_id": chunk.get("document_id"),
                    "document_name": chunk.get("document_name", "Unknown"),
                    "similarity_score": chunk.get("similarity_score", 0.0),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "metadata": chunk.get("metadata", {})
                }
                enhanced_context.append(context_chunk)
            
            logger.info(f"Retrieved {len(enhanced_context)} relevant chunks with aggressive search")
            return enhanced_context
            
        except Exception as e:
            logger.error(f"Error retrieving relevant context: {e}")
            return []
    
    def generate_rag_response(self, user_message: str, session_id: str, 
                             context_chunks: Optional[List[Dict[str, Any]]] = None,
                             storage_backend: str = 'pgvector') -> Dict[str, Any]:
        """
        Generate a comprehensive response using enhanced RAG with chat history
        """
        try:
            logger.info(f"Generating enhanced RAG response for session {session_id}")
            
            # Get chat history for context
            chat_history = self.get_chat_history(session_id, limit=self.max_chat_history)
            
            # Retrieve relevant context if not provided (now includes chat history)
            if context_chunks is None:
                # Use provided storage backend preference when retrieving context
                session_backend = storage_backend or getattr(settings, 'DEFAULT_STORAGE_BACKEND', 'pgvector')
                context_chunks = self.retrieve_relevant_context(user_message, session_id, storage_backend=session_backend)
            
            # Ensure context_chunks is not None
            if context_chunks is None:
                context_chunks = []
            
            # Build enhanced prompt with chat history and comprehensive context
            prompt = self._build_enhanced_prompt(user_message, context_chunks, chat_history)
            
            # Generate response using Ollama (sync call)
            response = ollama_client.generate_chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
                top_p=0.9,
                stream=False
            )
            
            if not response:
                logger.error("Ollama returned empty response")
                return {
                    "success": False,
                    "content": "I apologize, but I'm having trouble generating a response right now. Please try again.",
                    "error": "Failed to generate response from LLM"
                }
            
            # Store the conversation
            self.store_chat_message(session_id, "user", user_message)
            self.store_chat_message(session_id, "assistant", response, {
                "model": self.chat_model,
                "context_chunks_used": len(context_chunks),
                "chat_history_used": len(chat_history),
                "prompt_length": len(prompt)
            })
            
            return {
                "success": True,
                "content": response,
                "context_chunks": context_chunks,
                "chat_history_length": len(chat_history),
                "model_used": self.chat_model,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error generating enhanced RAG response: {e}")
            return {
                "success": False,
                "content": "I encountered an error while processing your request. Please try again.",
                "error": str(e)
            }
    
    def _build_enhanced_prompt(self, user_message: str, context_chunks: List[Dict[str, Any]], 
                              chat_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build an enhanced prompt with chat history and better context integration"""
        
        # Enhanced system prompt for longer, more detailed responses
        system_prompt = """You are an expert AI assistant with access to a comprehensive knowledge base. 
Your task is to provide detailed, informative, and well-structured responses using the provided context and conversation history.

RESPONSE GUIDELINES:
- Provide comprehensive answers with sufficient detail and examples
- Use the context documents to support your explanations
- Reference specific information from the knowledge base when relevant
- Maintain consistency with previous conversation context
- Structure your response clearly with proper organization
- Be thorough while remaining concise and focused
- Aim for responses that are informative and complete"""
        
        # Chat history section
        history_section = ""
        if chat_history and len(chat_history) > 0:
            history_section = "\n\nCONVERSATION HISTORY:\n"
            # Include last few exchanges for context
            recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
            for msg in recent_history:
                role = "USER" if msg.get("message_type") == "user" else "ASSISTANT"
                content = msg.get("content", "")[:200]  # Truncate for context
                history_section += f"{role}: {content}\n"
            history_section += "\n"
        
        # Enhanced context section with more comprehensive chunks
        context_section = ""
        if context_chunks:
            context_section = "\nKNOWLEDGE BASE CONTEXT:\n"
            for i, chunk in enumerate(context_chunks, 1):
                content = chunk.get("content", "")[:800]  # Longer chunks for better context
                doc_name = chunk.get("document_name", "Unknown")
                similarity = chunk.get("similarity_score", 0.0)
                context_section += f"[Source {i}] ({doc_name}, relevance: {similarity:.2f}):\n{content}\n\n"
        
        # Enhanced prompt structure
        prompt = f"""{system_prompt}{history_section}{context_section}

CURRENT USER QUESTION: {user_message}

Please provide a comprehensive and well-structured response based on the knowledge base context and conversation history. Include relevant details, examples, and explanations to fully address the user's question.

ASSISTANT:"""
        
        return prompt
    

    
    def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session from PostgreSQL - optimized sync version"""
        return self.delete_chat_session_sync(session_id)
    
    def get_user_sessions(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get all chat sessions for a user from PostgreSQL - optimized sync version"""
        try:
            with connection.cursor() as cursor:
                # For now, return all sessions since we don't have user_id in sessions table
                # This could be enhanced by adding user_id column to sessions
                cursor.execute("""
                    SELECT s.id, s.status, s.created_at, s.updated_at,
                           COUNT(c.id) as message_count
                    FROM sessions s
                    LEFT JOIN conversations c ON s.id = c.session_id
                    GROUP BY s.id, s.status, s.created_at, s.updated_at
                    ORDER BY s.updated_at DESC
                    LIMIT %s
                """, [limit])
                
                results = cursor.fetchall()
                sessions = []
                for row in results:
                    sessions.append({
                        'id': row[0],
                        'status': row[1],
                        'created_at': row[2],
                        'updated_at': row[3],
                        'message_count': row[4]
                    })
                
                return sessions
            
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []

    def get_available_models(self) -> List[str]:
        """Get available chat models - optimized sync version"""
        return self.get_available_models_sync()

    def get_available_models_sync(self) -> List[str]:
        """Get available chat models - sync wrapper"""
        try:
            model_status = ollama_client.check_model_availability()
            return model_status.get('models', [self.chat_model])
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return [self.chat_model]
    
    def get_chat_history_sync(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get chat history - optimized sync wrapper"""
        try:
            with connection.cursor() as cursor:
                # Optimized query with index hint
                cursor.execute("""
                    SELECT id, session_id, message_type, content, message_index, metadata, created_at
                    FROM conversations
                    WHERE session_id = %s
                    ORDER BY message_index ASC
                    LIMIT %s
                """, [session_id, limit])
                
                results = cursor.fetchall()
                messages = []
                for row in results:
                    messages.append({
                        'id': row[0],
                        'session_id': row[1],
                        'message_type': row[2],
                        'content': row[3],
                        'message_index': row[4],
                        'metadata': row[5],
                        'created_at': row[6]
                    })
                
                return messages
            
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []
    
    def delete_chat_session_sync(self, session_id: str) -> bool:
        """Delete a chat session - sync wrapper"""
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    DELETE FROM sessions
                    WHERE id = %s
                """, [session_id])
                
                deleted = cursor.rowcount > 0
            
            if deleted:
                logger.info(f"Deleted chat session: {session_id}")
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting chat session: {e}")
            return False
    
    def generate_rag_response_sync(self, user_message: str, session_id: str, 
                                  context_chunks: Optional[List[Dict[str, Any]]] = None,
                                  storage_backend: str = 'pgvector') -> Dict[str, Any]:
        """Generate RAG response - sync wrapper using sync database operations"""
        try:
            logger.info(f"Generating RAG response for session {session_id}")
            
            # Get chat history using sync method
            chat_history = self.get_chat_history_sync(session_id, limit=self.max_chat_history)
            
            # Retrieve relevant context if not provided
            if context_chunks is None:
                context_chunks = self.retrieve_relevant_context(user_message, session_id, storage_backend=storage_backend)
            
            # Ensure context_chunks is not None
            if context_chunks is None:
                context_chunks = []
            
            # Build enhanced prompt with chat history and comprehensive context
            prompt = self._build_enhanced_prompt(user_message, context_chunks, chat_history)
            
            # Generate response using Ollama (sync call)
            response = ollama_client.generate_chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
                top_p=0.9,
                stream=False
            )
            
            if not response:
                logger.error("Ollama returned empty response")
                return {
                    "success": False,
                    "content": "I apologize, but I'm having trouble generating a response right now. Please try again.",
                    "error": "Failed to generate response from LLM"
                }
            
            # Store the conversation using sync database operations
            self.store_chat_message_sync(session_id, "user", user_message)
            self.store_chat_message_sync(session_id, "assistant", response, {
                "model": self.chat_model,
                "context_chunks_used": len(context_chunks),
                "chat_history_used": len(chat_history)
            })
            
            return {
                "success": True,
                "content": response,
                "context_chunks": context_chunks,
                "chat_history_length": len(chat_history),
                "model_used": self.chat_model,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return {
                "success": False,
                "content": "I encountered an error while processing your request. Please try again.",
                "error": str(e)
            }
    
    def store_chat_message_sync(self, session_id: str, role: str, content: str, 
                               metadata: Optional[Dict[str, Any]] = None) -> int:
        """Store chat message - sync wrapper with session auto-creation"""
        try:
            with connection.cursor() as cursor:
                # Ensure session exists in sessions table (create if needed)
                cursor.execute("""
                    INSERT INTO sessions (id, status, created_at, updated_at)
                    VALUES (%s, 'active', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (id) DO NOTHING
                """, [session_id])
                
                # Get next message index for this session
                cursor.execute("""
                    SELECT COALESCE(MAX(message_index), -1) + 1
                    FROM conversations
                    WHERE session_id = %s
                """, [session_id])
                
                result = cursor.fetchone()
                message_index = result[0] if result else 0
                
                # Insert message (id is auto-generated by SERIAL)
                cursor.execute("""
                    INSERT INTO conversations (session_id, message_type, content, message_index, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    RETURNING id
                """, [session_id, role, content, message_index, json.dumps(metadata or {})])
                
                # Get the auto-generated message ID
                result = cursor.fetchone()
                message_id = result[0] if result else 0
                
                # Update session timestamp
                cursor.execute("""
                    UPDATE sessions
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, [session_id])
                
                logger.info(f"Stored message {message_id} in session {session_id}")
                return message_id
            
        except Exception as e:
            logger.error(f"Error storing chat message: {e}")
            return 0  # Return 0 for graceful degradation

# Global chat service instance
chat_service = ChatService()