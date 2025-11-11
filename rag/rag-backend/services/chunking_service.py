"""
Dedicated Chunking Service for PDF Processing
Handles all text chunking operations with word-based recursive semantic algorithm
"""
import logging
import time
import re
from typing import List, Dict, Any, Optional
import PyPDF2
import io

logger = logging.getLogger(__name__)


class ChunkingService:
    """
    Service dedicated to text chunking operations
    Provides word-based recursive semantic chunking optimized for PDFs
    """
    
    def __init__(self):
        """Initialize chunking service with configuration"""
        # Word-based configuration for fast processing
        self.target_words = 200  # ~1000 characters
        self.max_words = 300     # ~1500 characters
        self.min_words = 50      # ~250 characters
        self.overlap_words = 30  # ~150 characters overlap
        
        logger.info("ChunkingService initialized with word-based configuration")
        logger.info(f"  Target: {self.target_words} words, Max: {self.max_words}, Min: {self.min_words}")
    
    def extract_text_from_pdf(self, pdf_file) -> tuple[str, int, List[str]]:
        """
        Extract text from a PDF file
        Returns: (full_text, page_count, page_texts)
        """
        try:
            # Read PDF file
            pdf_content = pdf_file.read() if hasattr(pdf_file, 'read') else pdf_file
            pdf_file_obj = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
            
            page_count = len(pdf_reader.pages)
            page_texts = []
            full_text = []
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        page_texts.append(page_text)
                        full_text.append(page_text)
                    else:
                        logger.warning(f"Page {page_num} has no extractable text (may be image-based)")
                except Exception as e:
                    logger.error(f"Error extracting text from page {page_num}: {e}")
            
            combined_text = "\n\n".join(full_text)
            logger.info(f"Extracted {len(combined_text)} characters from {page_count} pages")
            
            return combined_text, page_count, page_texts
            
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return "", 0, []
    
    def word_based_recursive_chunking(
        self, 
        text: str, 
        filename: str = "",
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Fast word-based recursive semantic chunking optimized for PDF content
        
        Algorithm:
        1. Split by paragraphs (\\n\\n) - natural boundaries
        2. Combine paragraphs to reach target word count
        3. Split by sentences if paragraph too large
        4. Add overlap between chunks for context continuity
        
        Returns list of chunk dictionaries with detailed metadata
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for chunking: {filename}")
            return []
        
        start_time = time.time()
        logger.info(f"Starting word-based chunking: {len(text)} chars from {filename}")
        
        chunks = []
        metadata = metadata or {}
        
        # Preprocess text - clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        text = re.sub(r' {2,}', ' ', text)      # Max 1 space
        
        def split_into_chunks(text_segment: str) -> List[str]:
            """Split text into optimal word-based chunks"""
            # Split into paragraphs first
            paragraphs = [p.strip() for p in text_segment.split('\n\n') if p.strip()]
            
            if not paragraphs:
                return []
            
            result_chunks = []
            current_words = []
            
            for para in paragraphs:
                para_words = para.split()
                
                # If adding paragraph stays within target
                if len(current_words) + len(para_words) <= self.target_words:
                    if current_words:
                        current_words.append('\n\n')  # Keep paragraph separator
                    current_words.extend(para_words)
                
                # Current chunk is good size, save and start new
                elif len(current_words) >= self.min_words:
                    result_chunks.append(' '.join(current_words).replace(' \n\n ', '\n\n'))
                    
                    # Add overlap from previous chunk
                    if len(current_words) > self.overlap_words:
                        overlap = current_words[-self.overlap_words:]
                        current_words = overlap + ['\n\n'] + para_words
                    else:
                        current_words = para_words
                
                # Paragraph too large - split by sentences
                elif len(para_words) > self.max_words:
                    # Save current if exists
                    if current_words:
                        result_chunks.append(' '.join(current_words).replace(' \n\n ', '\n\n'))
                    
                    # Split large paragraph by sentences
                    sentences = self._split_into_sentences(para)
                    current_words = []
                    
                    for sentence in sentences:
                        sent_words = sentence.split()
                        
                        if len(current_words) + len(sent_words) <= self.target_words:
                            current_words.extend(sent_words)
                        elif len(current_words) >= self.min_words:
                            result_chunks.append(' '.join(current_words))
                            
                            # Add overlap
                            if len(current_words) > self.overlap_words:
                                current_words = current_words[-self.overlap_words:] + sent_words
                            else:
                                current_words = sent_words
                        else:
                            current_words.extend(sent_words)
                        
                        # Force split if too large
                        if len(current_words) > self.max_words:
                            # Split at max_words boundary
                            result_chunks.append(' '.join(current_words[:self.max_words]))
                            current_words = current_words[self.max_words - self.overlap_words:]
                
                else:
                    # Keep accumulating
                    if current_words:
                        current_words.append('\n\n')
                    current_words.extend(para_words)
                
                # Safety check: if accumulated too much, force split
                if len(current_words) > self.max_words * 1.5:
                    result_chunks.append(' '.join(current_words[:self.max_words]))
                    current_words = current_words[self.max_words - self.overlap_words:]
            
            # Add final chunk
            if current_words:
                final_text = ' '.join(current_words).replace(' \n\n ', '\n\n')
                if len(current_words) >= self.min_words or not result_chunks:
                    result_chunks.append(final_text)
                elif result_chunks:
                    result_chunks[-1] += '\n\n' + final_text
            
            return result_chunks
        
        # Create chunks
        chunk_texts = split_into_chunks(text.strip())
        
        # Convert to chunk dictionaries with rich metadata
        for idx, chunk_text in enumerate(chunk_texts):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            
            # Calculate metrics
            word_count = len(chunk_text.split())
            char_count = len(chunk_text)
            sentence_count = self._count_sentences(chunk_text)
            paragraph_count = chunk_text.count('\n\n') + 1
            
            # Quality check
            is_optimal = self.min_words <= word_count <= self.max_words
            
            chunk_dict = {
                'text': chunk_text,
                'content': chunk_text,  # Alias for compatibility
                'chunk_index': idx,
                'index': idx,
                'length': char_count,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'paragraph_count': paragraph_count,
                'type': 'word_based_semantic',
                'chunk_method': 'word_based_recursive_semantic',
                'filename': filename,
                'is_optimal_size': is_optimal,
                'metadata': {
                    **metadata,
                    'avg_words_per_sentence': round(word_count / max(1, sentence_count), 1),
                    'avg_chars_per_word': round(char_count / max(1, word_count), 1),
                    'density_score': round(word_count / max(1, char_count) * 100, 2)
                }
            }
            
            chunks.append(chunk_dict)
        
        # Calculate statistics
        processing_time = time.time() - start_time
        
        if chunks:
            avg_words = sum(c['word_count'] for c in chunks) / len(chunks)
            avg_chars = sum(c['length'] for c in chunks) / len(chunks)
            min_words = min(c['word_count'] for c in chunks)
            max_words = max(c['word_count'] for c in chunks)
            optimal_count = sum(1 for c in chunks if c['is_optimal_size'])
            
            logger.info(f"✅ Chunking complete in {processing_time:.2f}s: {len(chunks)} chunks")
            logger.info(f"   Words: avg={avg_words:.1f}, min={min_words}, max={max_words}, target={self.target_words}")
            logger.info(f"   Chars: avg={avg_chars:.1f}, Optimal: {optimal_count}/{len(chunks)} ({optimal_count/len(chunks)*100:.1f}%)")
        else:
            logger.warning(f"No chunks created from {len(text)} characters")
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = []
        current = []
        
        for word in text.split():
            current.append(word)
            # Check if word ends with sentence terminator
            if word.rstrip().endswith(('.', '!', '?')):
                sentences.append(' '.join(current))
                current = []
        
        if current:
            sentences.append(' '.join(current))
        
        return sentences
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text"""
        count = 0
        for char in ['.', '!', '?']:
            count += text.count(char)
        return max(1, count)
    
    def chunk_pdf_file(
        self, 
        pdf_file, 
        filename: str = None
    ) -> Dict[str, Any]:
        """
        Complete PDF chunking pipeline: extract text + chunk
        Returns detailed result with chunks and statistics
        """
        start_time = time.time()
        
        if filename is None:
            filename = getattr(pdf_file, 'filename', 'unknown.pdf')
        
        logger.info(f"Starting PDF chunking pipeline for: {filename}")
        
        # Extract text
        text_content, page_count, page_texts = self.extract_text_from_pdf(pdf_file)
        
        if not text_content:
            return {
                'success': False,
                'error': 'No text extracted from PDF',
                'filename': filename,
                'page_count': page_count
            }
        
        # Create chunks with page metadata
        metadata = {
            'page_count': page_count,
            'source_type': 'pdf'
        }
        
        chunks = self.word_based_recursive_chunking(text_content, filename, metadata)
        
        if not chunks:
            return {
                'success': False,
                'error': 'No chunks created',
                'filename': filename,
                'text_length': len(text_content)
            }
        
        # Calculate detailed statistics
        processing_time = time.time() - start_time
        
        chunk_lengths = [c['length'] for c in chunks]
        word_counts = [c['word_count'] for c in chunks]
        sentence_counts = [c['sentence_count'] for c in chunks]
        paragraph_counts = [c['paragraph_count'] for c in chunks]
        optimal_chunks = sum(1 for c in chunks if c['is_optimal_size'])
        
        result = {
            'success': True,
            'filename': filename,
            'page_count': page_count,
            'text_length': len(text_content),
            'chunks': chunks,
            'chunk_count': len(chunks),
            'processing_time': round(processing_time, 2),
            'processing_speed': {
                'chars_per_second': round(len(text_content) / processing_time, 0),
                'chunks_per_second': round(len(chunks) / processing_time, 1),
                'pages_per_second': round(page_count / processing_time, 2)
            },
            'chunk_statistics': {
                'total_chunks': len(chunks),
                'optimal_chunks': optimal_chunks,
                'optimal_ratio': round(optimal_chunks / len(chunks) * 100, 1),
                'avg_size_chars': round(sum(chunk_lengths) / len(chunks), 1),
                'min_size_chars': min(chunk_lengths),
                'max_size_chars': max(chunk_lengths),
                'avg_words': round(sum(word_counts) / len(chunks), 1),
                'min_words': min(word_counts),
                'max_words': max(word_counts),
                'avg_sentences': round(sum(sentence_counts) / len(chunks), 1),
                'avg_paragraphs': round(sum(paragraph_counts) / len(chunks), 1),
                'total_words': sum(word_counts),
                'total_characters': sum(chunk_lengths)
            },
            'quality_assessment': {
                'grade': 'Excellent' if optimal_chunks / len(chunks) >= 0.8 else 'Good' if optimal_chunks / len(chunks) >= 0.6 else 'Fair',
                'recommendations': self._generate_recommendations(chunks)
            }
        }
        
        return result
    
    def _generate_recommendations(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Generate quality recommendations based on chunk analysis"""
        recommendations = []
        
        if not chunks:
            return ["No chunks to analyze"]
        
        avg_words = sum(c['word_count'] for c in chunks) / len(chunks)
        optimal_count = sum(1 for c in chunks if c['is_optimal_size'])
        optimal_ratio = optimal_count / len(chunks)
        
        if optimal_ratio >= 0.9:
            recommendations.append("✅ Excellent chunk distribution - ready for embedding")
        elif optimal_ratio >= 0.7:
            recommendations.append("✅ Good chunk quality - suitable for RAG")
        else:
            recommendations.append("⚠️ Some chunks may be suboptimal - consider adjusting parameters")
        
        if avg_words < 100:
            recommendations.append("💡 Average chunk size is small - might want to increase target words")
        elif avg_words > 250:
            recommendations.append("💡 Average chunk size is large - might want to decrease target words")
        
        return recommendations
    
    def get_chunk_preview(self, chunks: List[Dict[str, Any]], count: int = 5) -> List[Dict[str, Any]]:
        """Get preview of first N chunks with truncated text"""
        previews = []
        
        for chunk in chunks[:count]:
            preview = {
                'chunk_index': chunk['chunk_index'],
                'text_preview': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                'word_count': chunk['word_count'],
                'length': chunk['length'],
                'is_optimal': chunk['is_optimal_size']
            }
            previews.append(preview)
        
        return previews

# Global instance
chunking_service = ChunkingService()
