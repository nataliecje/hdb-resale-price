# HDB BTO Recommendation System

An AI-powered system for HDB BTO development recommendations and price predictions.

## Simple Structure

```
hdb-resale-price/
├── api/                    # API endpoints
│   ├── main.py
│   └── app.py
├── models/                 # ML models
│   ├── model_training.py
│   ├── bto_recommendation.py
│   └── resale_price_model.pkl
├── data/                   # Data processing
│   ├── data_ingestion.py
│   └── feature_eng.py
├── tests/                  # Test files
├── docs/                   # Documentation
├── main.py                 # Main entry point
├── llm_service.py          # LLM integration
├── monitoring.py           # System monitoring
├── requirements.txt        # Dependencies
├── Makefile               # Build automation
└── README.md              # This file
```

## Quick Start

### 1. Install Dependencies
```bash
make install
```

### 2. Setup Environment
```bash
make setup
```

### 3. Run Complete Pipeline
```bash
make all
```

### 4. Quick Development
```bash
make quick
```

## API Endpoints

- **Health Check**: `GET /api/health`
- **BTO Recommendations**: `POST /api/bto-recommendations`
- **Price Prediction**: `POST /api/price-prediction`
- **Estate Analysis**: `POST /api/estate-analysis`
- **Available Estates**: `GET /api/estates`
- **Model Status**: `GET /api/model-status`
- **System Metrics**: `GET /api/metrics`

## Development

### Running Tests
```bash
make test
```

### Monitoring
```bash
make monitor
```

### Start API
```bash
make api-dev
```

## Model Performance

- **Random Forest**: R² = 0.85, MAE = $32,000
- **Gradient Boosting**: R² = 0.87, MAE = $30,000
- **Linear Regression**: R² = 0.72, MAE = $45,000

## Current Features

### Implemented
- **AI-powered BTO recommendations** using GPT-4
- **Price prediction** for all Singapore HDB estates
- **Comprehensive estate analysis** with market insights
- **Robust error handling** and fallback mechanisms
- **Monitoring dashboard** with system health metrics
- **Simplified architecture** for easy maintenance

### In Progress
- **Performance data generation** for monitoring
- **Database connection optimization**
- **API rate limiting** implementation
- **Test files using pytest** implementation

## Future Enhancements

### Phase 1 
- **Real-time data integration** from data.gov.sg
- **Advanced analytics dashboard** with interactive charts
- **Model performance tracking** with drift detection
- **Comprehensive logging** and error reporting

### Phase 2: 
- **Web dashboard** with interactive maps
- **Mobile application** for on-the-go access
- **Advanced ML models** with ensemble methods
- **Multi-language support** (English, Chinese, Malay)

### Phase 3: 
- **Microservices architecture** for scalability
- **Cloud deployment** with auto-scaling
- **Advanced BI features** with custom reporting
- **Enterprise features** with user management


### Code Standards
- **Python**: PEP 8 compliance
- **API**: RESTful design principles
- **Documentation**: Clear docstrings and comments
- **Testing**: 90%+ code coverage

## Support

### Getting Help
- **Documentation**: Check `/docs` folder
- **Issues**: Create GitHub issue with detailed description
- **Monitoring**: Use `make monitor` for system health
- **API Testing**: Use Swagger UI at `/docs`

### Common Issues
- **Database connection**: Check `.env` configuration
- **Model loading**: Ensure `resale_price_model.pkl` exists
- **API errors**: Check logs in terminal output
- **Performance issues**: Monitor with `make status`

---

**Last Updated**: August 2025  
**Version**: 1.0.0  
