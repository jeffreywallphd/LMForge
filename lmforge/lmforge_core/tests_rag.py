"""
Unit tests for RAG Chat views
"""
from django.test import TestCase, Client
from django.urls import reverse
from lmforge_core.models.rag_chat import ChatSession, ChatMessage
import json


class ChatSessionModelTest(TestCase):
    """Test ChatSession model"""

    def setUp(self):
        self.session = ChatSession.objects.create(session_id="test-session-123")

    def test_session_creation(self):
        """Test that session is created correctly"""
        self.assertEqual(self.session.session_id, "test-session-123")
        self.assertIsNotNone(self.session.created_at)
        self.assertIsNotNone(self.session.updated_at)

    def test_session_str(self):
        """Test session string representation"""
        self.assertEqual(str(self.session), "Session test-session-123")


class ChatMessageModelTest(TestCase):
    """Test ChatMessage model"""

    def setUp(self):
        self.session = ChatSession.objects.create(session_id="test-session-456")
        self.message = ChatMessage.objects.create(
            session=self.session,
            message="Hello, how are you?",
            response="I'm doing well, thank you!"
        )

    def test_message_creation(self):
        """Test that message is created correctly"""
        self.assertEqual(self.message.message, "Hello, how are you?")
        self.assertEqual(self.message.response, "I'm doing well, thank you!")
        self.assertEqual(self.message.session, self.session)

    def test_message_ordering(self):
        """Test that messages are ordered by timestamp"""
        message2 = ChatMessage.objects.create(
            session=self.session,
            message="Second message"
        )
        messages = ChatMessage.objects.all()
        self.assertEqual(messages[0], self.message)
        self.assertEqual(messages[1], message2)


class RAGChatViewTest(TestCase):
    """Test RAG Chat view"""

    def setUp(self):
        self.client = Client()
        self.url = reverse('rag-chat-view')

    def test_rag_chat_view_loads(self):
        """Test that RAG chat view loads successfully"""
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'rag_chat.html')

    def test_session_is_created_on_first_visit(self):
        """Test that a session is created on first visit"""
        response = self.client.get(self.url)
        # Session should be created via get_or_create_session
        self.assertIn('session_id', response.context)
