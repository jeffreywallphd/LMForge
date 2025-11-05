from django.db import models
import json

class DocumentChunk(models.Model):
    """Model to store document chunks with embeddings for RAG"""
    text = models.TextField()
    embedding = models.JSONField()  # Store embedding as JSON array
    collection_name = models.CharField(max_length=255, default="pdf_embeddings", db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'pdf_embeddings'  # Use same table name
        indexes = [
            models.Index(fields=['collection_name']),
        ]
    
    def __str__(self):
        return f"Chunk {self.id} - {self.text[:50]}..."
    
    def get_embedding_array(self):
        """Convert JSON embedding to numpy array"""
        import numpy as np
        if isinstance(self.embedding, list):
            return np.array(self.embedding)
        elif isinstance(self.embedding, str):
            return np.array(json.loads(self.embedding))
        return np.array(self.embedding)

