# Deployment Guide

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment:
   ```bash
   cp config/.env.example config/.env
   # Edit config/.env with your settings
   ```

3. Run the application:
   ```bash
   python src/main.py
   ```

## Production Deployment

1. Build Docker image:
   ```bash
   docker build -t hdb-bto-system .
   ```

2. Run container:
   ```bash
   docker run -p 8000:8000 hdb-bto-system
   ```

## Environment Variables

- `DB_USER`: Database username
- `DB_PASSWORD`: Database password
- `DB_HOST`: Database host
- `DB_PORT`: Database port
- `DB_NAME`: Database name
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
