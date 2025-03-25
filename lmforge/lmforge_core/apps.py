from django.apps import AppConfig


class LmforgeCoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'lmforge_core'

    def ready(self):
        import lmforge_core.signals
