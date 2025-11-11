from django.db import models


class ChatSession(models.Model):
    """Model to track chat sessions"""
    session_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Session {self.session_id}"


class ProcessedDocument(models.Model):
    """Model to track processed documents"""
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE)
    filename = models.CharField(max_length=255)
    file_size = models.IntegerField()
    chunks_created = models.IntegerField(default=0)
    chunking_method = models.CharField(max_length=50, default='semantic')
    processed_at = models.DateTimeField(auto_now_add=True)
    backend_session_id = models.CharField(max_length=100, blank=True, null=True)
    
    def __str__(self):
        return f"{self.filename} ({self.chunking_method})"


class ChatMessage(models.Model):
    """Model to store chat messages"""
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField(blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['timestamp']
    
    def __str__(self):
        return f"Message in {self.session.session_id}"
