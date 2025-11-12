# LMForge Production Deployment Checklist

## Pre-Deployment

- [ ] **Security**
  - [ ] Change `SECRET_KEY` in `.env`
  - [ ] Set `DEBUG=False`
  - [ ] Update `ALLOWED_HOSTS` with your domain
  - [ ] Update `CORS_ALLOWED_ORIGINS` with actual domains
  - [ ] Use HTTPS in production URLs

- [ ] **Database**
  - [ ] Set up PostgreSQL on production server
  - [ ] Update `DATABASE_*` credentials
  - [ ] Run `python manage.py migrate`
  - [ ] Create superuser: `python manage.py createsuperuser`

- [ ] **Environment**
  - [ ] Update all API keys (HF, OpenAI, Weights & Biases)
  - [ ] Configure external services (Qdrant if used)
  - [ ] Set up proper logging directories

- [ ] **Static Files**
  - [ ] Run `python manage.py collectstatic --noinput`
  - [ ] Configure web server (Nginx/Apache) to serve static files

- [ ] **Docker**
  - [ ] Update docker-compose for production
  - [ ] Use production-grade environment variables
  - [ ] Configure resource limits appropriately

## Deployment

- [ ] Use a production WSGI server (Gunicorn, uWSGI)
- [ ] Set up reverse proxy (Nginx)
- [ ] Configure SSL/TLS certificates
- [ ] Set up monitoring and logging
- [ ] Create database backups strategy
- [ ] Document access URLs and credentials (securely)

## Post-Deployment

- [ ] Test all endpoints
- [ ] Monitor logs for errors
- [ ] Set up automated backups
- [ ] Configure alerting for critical errors
- [ ] Document any custom configurations

## Production WSGI Server Example (Gunicorn)

```bash
gunicorn lmforge.wsgi:application --workers 4 --bind 0.0.0.0:8000 --timeout 120
```

## Production Docker Compose Example

```yaml
# Update image versions to specific tags
# Set environment variables from .env file
# Use named volumes for data persistence
# Configure proper health checks
```
