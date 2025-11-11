"""
Chat Service for RAG-based LLM Interactions
Handles chat sessions, context retrieval, and LLM communication
UPDATED: Uses PostgreSQL/pgVector directly - NO REDIS DEPENDENCY
"""
import logging
import uuid
import time
from typing import List, Dict, Any, Optional
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import os

from clients.ollama_client import ollama_client
from services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

class ChatService:
    
    
    def __init__(self):
        self.chat_model = "qwen2.5:0.5b-instruct"  # Fixed chat model
        self.default_model = self.chat_model  # Alias for API compatibility
        self.max_context_chunks = 8  # Increased for more aggressive search
        self.max_chat_history = 10  # Increased for better context
        self.similarity_threshold = 0.6  # Lower threshold for more results
        
        # Database connection parameters
        self.db_params = {
            'host': os.getenv("DB_HOST", "localhost"),
            'port': int(os.getenv("DB_PORT", 5432)),
            'database': os.getenv("DB_NAME", "pdf_rag_db"),
            'user': os.getenv("DB_USER", "pdf_rag_user"),
            'password': os.getenv("DB_PASSWORD", "pdf_rag_password")
        }
        
        logger.info("ChatService initialized - aggressive search with expanded context")
    
    def _get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_params)
        
    async def create_chat_session(self, user_id: str = None, session_name: str = None) -> str:
        """
        Create a new chat session in PostgreSQL
        Args:
            user_id: Optional user identifier
            session_name: Optional session name
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Insert session into PostgreSQL
            cursor.execute("""
                INSERT INTO sessions (id, status, created_at, updated_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (session_id, 'active'))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Created chat session in PostgreSQL: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating chat session: {e}")
            return session_id  # Return anyway for graceful degradation
    
    async def get_chat_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get chat session data from PostgreSQL"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT id, status, created_at, updated_at
                FROM sessions
                WHERE id = %s
            """, (session_id,))
            
            session = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if session:
                return dict(session)
            return None
            
        except Exception as e:
            logger.error(f"Error getting chat session: {e}")
            return None
    
    async def get_chat_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get chat history for a session from PostgreSQL
        Args:
            session_id: Chat session ID
            limit: Maximum number of messages to retrieve
        Returns:
            List of chat messages
        """
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT id, session_id, message_type, content, message_index, metadata, created_at
                FROM conversations
                WHERE session_id = %s
                ORDER BY message_index ASC
                LIMIT %s
            """, (session_id, limit))
            
            messages = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [dict(msg) for msg in messages]
            
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []
    
    async def store_chat_message(self, session_id: str, role: str, content: str, 
                                 metadata: Dict[str, Any] = None) -> int:
        """
        Store a chat message in PostgreSQL
        Args:
            session_id: Chat session ID
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Additional metadata
        Returns:
            Message ID (integer)
        """
        message_id = None
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Get next message index for this session
            cursor.execute("""
                SELECT COALESCE(MAX(message_index), -1) + 1
                FROM conversations
                WHERE session_id = %s
            """, (session_id,))
            
            message_index = cursor.fetchone()[0]
            
            # Insert message (id is auto-generated by SERIAL)
            cursor.execute("""
                INSERT INTO conversations (session_id, message_type, content, message_index, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                RETURNING id
            """, (session_id, role, content, message_index, json.dumps(metadata or {})))
            
            # Get the auto-generated message ID
            message_id = cursor.fetchone()[0]
            
            # Update session timestamp
            cursor.execute("""
                UPDATE sessions
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (session_id,))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Stored message {message_id} in session {session_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Error storing chat message: {e}")
            if message_id:
                return message_id
            return 0  # Return 0 for graceful degradation
    
    async def retrieve_relevant_context(self, query: str, session_id: str = None, 
                                       top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from the knowledge base using aggressive RAG search"""
        top_k = top_k or self.max_context_chunks
        
        try:
            # Get recent chat history for enhanced query context
            chat_history = await self.get_chat_history(session_id, limit=self.max_chat_history) if session_id else []
            
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
            
            # 1. Primary search with enhanced query
            primary_chunks = await embedding_service.search_similar_chunks(
                query_embedding, 
                top_k=top_k,
                threshold=self.similarity_threshold
            )
            all_chunks.extend(primary_chunks)
            
            # 2. Secondary search with original query (if different)
            if enhanced_query != query:
                original_embedding = ollama_client.generate_embedding(query)
                if original_embedding:
                    secondary_chunks = await embedding_service.search_similar_chunks(
                        original_embedding,
                        top_k=top_k//2,
                        threshold=self.similarity_threshold - 0.1  # Lower threshold
                    )
                    all_chunks.extend(secondary_chunks)
            
            # 3. Keyword-based fallback search (if few results)
            if len(all_chunks) < top_k//2:
                # TODO: Implement keyword search in database
                pass
            
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
    
    async def generate_rag_response(self, user_message: str, session_id: str, 
                                   context_chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive response using enhanced RAG with chat history
        """
        try:
            logger.info(f"Generating enhanced RAG response for session {session_id}")
            
            # Get chat history for context
            chat_history = await self.get_chat_history(session_id, limit=self.max_chat_history)
            
            # Retrieve relevant context if not provided (now includes chat history)
            if context_chunks is None:
                context_chunks = await self.retrieve_relevant_context(user_message, session_id)
            
            # Build enhanced prompt with chat history and comprehensive context
            prompt = self._build_enhanced_prompt(user_message, context_chunks, chat_history)
            
            # Generate response using Ollama with enhanced settings
            response = await self._call_ollama_generate_enhanced(prompt)
            
            if not response:
                logger.error("Ollama returned empty response")
                return {
                    "success": False,
                    "content": "I apologize, but I'm having trouble generating a response right now. Please try again.",
                    "error": "Failed to generate response from LLM"
                }
            
            # Store the conversation
            await self.store_chat_message(session_id, "user", user_message)
            await self.store_chat_message(session_id, "assistant", response["response"], {
                "model": self.chat_model,
                "context_chunks_used": len(context_chunks),
                "chat_history_used": len(chat_history),
                "generation_time": response.get("generation_time", 0),
                "prompt_length": len(prompt)
            })
            
            return {
                "success": True,
                "content": response["response"],
                "context_chunks": context_chunks,
                "chat_history_length": len(chat_history),
                "model_used": self.chat_model,
                "generation_time": response.get("generation_time", 0),
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
                              chat_history: List[Dict[str, Any]] = None) -> str:
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
    
    async def _call_ollama_generate(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Ollama generate API - simplified"""
        try:
            start_time = time.time()
            
            # Convert prompt to messages format
            messages = [{"role": "user", "content": prompt}]
            
            response_text = ollama_client.generate_chat_completion(messages)
            
            if not response_text:
                return None
            
            generation_time = time.time() - start_time
            
            return {
                "response": response_text,
                "model": self.chat_model,
                "generation_time": generation_time
            }
            
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return None
    
    async def _call_ollama_generate_enhanced(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Ollama generate API with enhanced settings for comprehensive responses"""
        try:
            start_time = time.time()
            
            # Convert prompt to messages format
            messages = [{"role": "user", "content": prompt}]
            
            # Enhanced generation with parameters for longer, more detailed responses
            response_text = ollama_client.generate_chat_completion(
                messages,
                temperature=0.3,  # Slightly creative but focused
                max_tokens=1024,  # Allow for longer responses
                top_p=0.9,       # Good diversity
                stream=False     # Get complete response
            )
            
            if not response_text:
                logger.warning("Enhanced Ollama generation returned empty response")
                return None
            
            generation_time = time.time() - start_time
            
            # Log response quality metrics
            response_length = len(response_text)
            logger.info(f"Enhanced response generated: {response_length} chars in {generation_time:.2f}s")
            
            return {
                "response": response_text,
                "model": self.chat_model,
                "generation_time": generation_time,
                "response_length": response_length,
                "enhanced": True
            }
            
        except Exception as e:
            logger.error(f"Error calling enhanced Ollama generation: {e}")
            # Fallback to regular generation
            return await self._call_ollama_generate(prompt)
    
    async def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session from PostgreSQL (cascades to conversations)"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Delete session (conversations will cascade)
            cursor.execute("""
                DELETE FROM sessions
                WHERE id = %s
            """, (session_id,))
            
            conn.commit()
            deleted = cursor.rowcount > 0
            cursor.close()
            conn.close()
            
            if deleted:
                logger.info(f"Deleted chat session: {session_id}")
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting chat session: {e}")
            return False
    
    async def get_user_sessions(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get all chat sessions for a user from PostgreSQL"""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
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
            """, (limit,))
            
            sessions = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [dict(session) for session in sessions]
            
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []

# Global chat service instance
chat_service = ChatService()