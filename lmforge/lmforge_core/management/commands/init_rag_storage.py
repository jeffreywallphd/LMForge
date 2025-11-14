"""
Initialize RAG Vector Storage - Django Management Command

Usage:
python manage.py init_rag_storage [--force] [--cpu]
"""
from django.core.management.base import BaseCommand
from lmforge_core.services.rag_vector_initializer import RAGVectorInitializer


class Command(BaseCommand):
    """Django management command for initializing RAG vector storage"""
    
    help = 'Initialize RAG vector storage from JSON files in media/JSON directory'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force reinitialization even if storage exists'
        )
        parser.add_argument(
            '--cpu',
            action='store_true',
            help='Use CPU instead of GPU for embeddings'
        )
    
    def handle(self, *args, **options):
        """Handle the management command"""
        self.stdout.write(
            self.style.SUCCESS('Starting RAG vector storage initialization...')
        )
        
        # Create initializer
        initializer = RAGVectorInitializer(self.stdout, self.style)
        
        # Run initialization
        result = initializer.initialize(
            force=options['force'],
            use_gpu=not options['cpu']
        )
        
        # Print results
        if result['success']:
            if result.get('already_exists'):
                self.stdout.write(
                    self.style.WARNING('Vector storage already initialized')
                )
                self.stdout.write('Use --force to reinitialize')
            else:
                self.stdout.write(
                    self.style.SUCCESS('Initialization successful!')
                )
                self.stdout.write(f"Files processed: {result['files_processed']}/{result['total_files']}")
                self.stdout.write(f"Total chunks: {result['total_chunks']}")
                self.stdout.write(f"Time elapsed: {result['elapsed_time']:.2f}s")
        else:
            self.stdout.write(
                self.style.ERROR(f"Initialization failed: {result.get('message', 'Unknown error')}")
            )
