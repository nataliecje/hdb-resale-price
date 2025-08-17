# HDB BTO Recommendation System Makefile
# Usage: make <target>

.PHONY: help install setup data train api monitor test clean all

# Default target
help:
	@echo "HDB BTO Recommendation System - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install     - Install Python dependencies"
	@echo "  make setup       - Set up environment"
	@echo ""
	@echo "Data Processing:"
	@echo "  make data        - Run data ingestion from data.gov.sg"
	@echo "  make features    - Run feature engineering"
	@echo ""
	@echo "Model Training:"
	@echo "  make train       - Train all models"
	@echo "  make train-fast  - Train models without hyperparameter tuning"
	@echo ""
	@echo "API & Services:"
	@echo "  make api         - Start FastAPI server"
	@echo "  make api-dev     - Start API server in development mode"
	@echo ""
	@echo "Monitoring & Testing:"
	@echo "  make monitor     - Run system monitoring"
	@echo "  make test        - Run automated tests"
	@echo "  make health      - Check system health"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean       - Clean generated files"
	@echo "  make all         - Run complete pipeline"
	@echo "  make status      - Show system status"

# Installation and Setup
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully"

setup:
	@echo "Setting up environment..."
	@if [ ! -f .env ]; then \
		echo "Creating .env file..."; \
		echo "OPENAI_API_KEY=your_openai_api_key" > .env; \
		echo "ANTHROPIC_API_KEY=your_anthropic_api_key" >> .env; \
		echo "DB_USER=postgres" >> .env; \
		echo "DB_PASSWORD=mysecretpassword" >> .env; \
		echo "DB_HOST=localhost" >> .env; \
		echo "DB_PORT=5432" >> .env; \
		echo "DB_NAME=hdb" >> .env; \
		echo "Please update API keys in .env"; \
	else \
		echo ".env file already exists"; \
	fi
	@echo "Environment setup complete"

# Data Processing
data:
	@echo "Starting data ingestion from data.gov.sg..."
	python data/data_ingestion.py
	@echo "Data ingestion complete"

features:
	@echo "Running feature engineering..."
	python data/feature_eng.py
	@echo "Feature engineering complete"

# Model Training
train:
	@echo "Training models with hyperparameter tuning..."
	python models/model_training.py
	@echo "Model training complete"

train-fast:
	@echo "Training models (fast mode - no hyperparameter tuning)..."
	@sed -i.bak 's/if best_model == "Random Forest":/if False and best_model == "Random Forest":/' models/model_training.py
	@python models/model_training.py
	@mv models/model_training.py.bak models/model_training.py
	@echo "Fast model training complete"

# API Server
api:
	@echo "Starting FastAPI server..."
	python main.py

api-dev:
	@echo "Starting FastAPI server in development mode..."
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --log-level info

# Monitoring and Testing
monitor:
	@echo "Running system monitoring..."
	python monitoring.py
	@echo "Monitoring complete"

test:
	@echo "Running automated tests..."
	pytest tests/ -v

health:
	@echo "Checking system health..."
	@if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then \
		echo "API server is running"; \
		curl -s http://localhost:8000/api/health | python -m json.tool; \
	else \
		echo "API server is not running. Start with: make api"; \
	fi

# Utilities
clean:
	@echo "Cleaning generated files..."
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	rm -f *.pkl *.csv *.png *.json
	@echo "Cleanup complete"

status:
	@echo "System Status:"
	@echo "=============="
	@echo "Python files:"
	@find . -name "*.py" | wc -l | xargs echo "  Total:"
	@echo "Data Files:"
	@find data/ -name "*.csv" 2>/dev/null | wc -l | xargs echo "  CSV files:"
	@echo "Model Files:"
	@find models/ -name "*.pkl" | wc -l | xargs echo "  Pickle files:"
	@echo ""
	@echo "Environment:"
	@if [ -f .env ]; then echo "  .env file exists"; else echo "  .env file missing"; fi
	@if command -v python3 > /dev/null; then echo "  Python3 available"; else echo "  Python3 not found"; fi
	@echo ""
	@echo "API Status:"
	@if curl -s http://localhost:8000/ > /dev/null 2>&1; then \
		echo "  API server running on http://localhost:8000"; \
	else \
		echo "  API server not running"; \
	fi

# Complete Pipeline
all: install setup data features train api
	@echo "Complete pipeline finished!"
	@echo "API available at: http://localhost:8000"
	@echo "Documentation at: http://localhost:8000/docs"

# Quick Start (for development)
quick: install setup features train-fast api-dev
	@echo "Quick start complete!"
	@echo "API available at: http://localhost:8000"
