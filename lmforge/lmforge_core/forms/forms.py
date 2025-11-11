from django import forms
from ..models.Document_Y import Document_y

class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document_y
        fields = ['pdf_file', 'large_text']




class DocumentProcessingForm(forms.Form):
    TEST_CHOICES = [
        ('real', 'Real Test'),
        ('mockup', 'Mock-up Test'),
    ]

    test_type = forms.ChoiceField(
        choices=TEST_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'}),
        label="Test Type"
    )

    instruction_prompt = forms.CharField(
        label="Instruction Prompt",
        required=False,
        widget=forms.Textarea(
            attrs={
                'rows': 5,
                'cols': 35,
                'placeholder': 'Add an instruction prompt to explain how the language model should behave'
            }
        )
    )

    num_paragraphs = forms.IntegerField(
        label="Number of Paragraphs",
        min_value=1,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

    num_questions = forms.IntegerField(
        label="Number of Questions",
        min_value=1,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

class MultipleFileInput(forms.ClearableFileInput):
    """Custom widget for multiple file uploads"""
    allow_multiple_selected = True

class MultipleFileField(forms.FileField):
    """Field for handling multiple file uploads"""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = [single_file_clean(data, initial)]
        return result

class PDFUploadForm(forms.Form):
    """Form for uploading PDF files - Step 1: Chunking only"""
    pdf_files = MultipleFileField(
        widget=MultipleFileInput(attrs={
            'accept': '.pdf',
            'class': 'form-control',
            'multiple': True,
            'id': 'pdf-files-input'
        }),
        help_text='Select one or more PDF files to upload (hold Ctrl/Cmd to select multiple)'
    )

class EmbeddingProcessForm(forms.Form):
    """Form for processing embeddings - Step 2: Generate and store embeddings"""
    use_gpu = forms.BooleanField(
        initial=True,
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text='Use GPU for embedding generation (if available)'
    )
    
    session_id = forms.CharField(
        widget=forms.HiddenInput(),
        required=True,
        help_text='Session ID from chunking step'
    )

class JSONDataForm(forms.Form):
    """Form for processing JSON data"""
    json_data = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 10,
            'class': 'form-control',
            'placeholder': 'Paste your JSON data here...\n\nExample:\n{\n  "documents": [\n    {"text": "Your document content here"},\n    {"text": "Another document"}\n  ]\n}',
            'style': 'font-family: monospace;'
        }),
        help_text='Enter JSON data to process'
    )
    
    chunking_method = forms.ChoiceField(
        choices=[
            ('semantic', 'Semantic Based Chunking'),
            ('recursive', 'Recursive Character Chunking'),
            ('document_specific', 'Document Specific Chunking'),
            ('semantic_embedding', 'Semantic Embedding Chunking'),
        ],
        initial='semantic',
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text='Choose the chunking method for processing'
    )
    
    chunk_size = forms.IntegerField(
        initial=200,
        min_value=50,
        max_value=1000,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Maximum chunk size in characters'
    )
    
    chunk_overlap = forms.IntegerField(
        initial=50,
        min_value=0,
        max_value=200,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        help_text='Overlap between chunks in characters'
    )

class QuestionForm(forms.Form):
    """Form for asking questions"""
    question = forms.CharField(
        max_length=500,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Ask a question about your documents...',
            'autocomplete': 'off'
        }),
        help_text='Ask a question about the uploaded documents'
    )

class ChatMessageForm(forms.Form):
    """Form for chat messages"""
    message = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 2,
            'placeholder': 'Type your message here...',
            'id': 'chat-message-input',
            'style': 'resize: none;'
        }),
        label='',
        max_length=2000,
        required=True
    )
    
    session_id = forms.CharField(
        widget=forms.HiddenInput(),
        required=False
    )
    
    model = forms.CharField(
        widget=forms.HiddenInput(),
        required=False,
        initial='llama3.2'
    )
    
    use_rag = forms.BooleanField(
        widget=forms.HiddenInput(),
        required=False,
        initial=True
    )

class ChatSessionForm(forms.Form):
    """Form for creating chat sessions"""
    session_name = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter session name (optional)'
        }),
        label='Session Name',
        max_length=100,
        required=False
    )
    
    user_id = forms.CharField(
        widget=forms.HiddenInput(),
        required=False,
        initial='anonymous'
    )