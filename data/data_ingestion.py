import requests
import pandas as pd
import json
from sqlalchemy import create_engine, text
import time
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_USER = 'postgres'
DB_PASSWORD = 'mysecretpassword'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'hdb'

# Data.gov.sg API configuration
BASE_URL = "https://data.gov.sg/api/action/datastore_search"
RESOURCE_ID = "f1765b54-a209-4711-9761-2c5c2b5c2b5c"  # HDB Resale Price Index
LIMIT = 1000  # Records per request

class HDBDataIngestion:
    def __init__(self):
        self.engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
        self.session = requests.Session()
        
    def fetch_hdb_data(self, start_date=None, end_date=None):
        """
        Fetch HDB resale transaction data from data.gov.sg
        """
        logger.info("Starting HDB data ingestion from data.gov.sg")
        
        # Default to last 5 years if no dates specified
        if not start_date:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        all_data = []
        offset = 0
        
        while True:
            try:
                # API request parameters
                params = {
                    'resource_id': RESOURCE_ID,
                    'limit': LIMIT,
                    'offset': offset,
                    'filters': json.dumps({
                        'month': f"{start_date}:{end_date}"
                    })
                }
                
                response = self.session.get(BASE_URL, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if not data.get('result', {}).get('records'):
                    break
                    
                records = data['result']['records']
                all_data.extend(records)
                
                logger.info(f"Fetched {len(records)} records (offset: {offset})")
                
                # Check if we've reached the end
                if len(records) < LIMIT:
                    break
                    
                offset += LIMIT
                time.sleep(0.1)  # Rate limiting
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching data: {e}")
                break
                
        logger.info(f"Total records fetched: {len(all_data)}")
        return all_data
    
    def process_raw_data(self, raw_data):
        """
        Process and clean raw HDB data
        """
        if not raw_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(raw_data)
        
        # Standardize column names
        column_mapping = {
            'month': 'month',
            'town': 'town',
            'flat_type': 'flat_type',
            'block': 'block',
            'street_name': 'street_name',
            'storey_range': 'storey_range',
            'floor_area_sqm': 'floor_area_sqm',
            'flat_model': 'flat_model',
            'lease_commence_date': 'lease_commence_date',
            'remaining_lease': 'remaining_lease',
            'resale_price': 'resale_price'
        }
        
        # Rename columns that exist
        existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_columns)
        
        # Convert data types
        if 'resale_price' in df.columns:
            df['resale_price'] = pd.to_numeric(df['resale_price'], errors='coerce')
        if 'floor_area_sqm' in df.columns:
            df['floor_area_sqm'] = pd.to_numeric(df['floor_area_sqm'], errors='coerce')
        if 'lease_commence_date' in df.columns:
            df['lease_commence_date'] = pd.to_numeric(df['lease_commence_date'], errors='coerce')
            
        # Remove rows with missing critical data
        df = df.dropna(subset=['resale_price', 'town', 'flat_type'])
        
        logger.info(f"Processed data shape: {df.shape}")
        return df
    
    def create_database_schema(self):
        """
        Create database schema for HDB data
        """
        schema_sql = """
        CREATE TABLE IF NOT EXISTS resale_transactions (
            id SERIAL PRIMARY KEY,
            month VARCHAR(10),
            town VARCHAR(50),
            flat_type VARCHAR(20),
            block VARCHAR(10),
            street_name VARCHAR(100),
            storey_range VARCHAR(20),
            floor_area_sqm DECIMAL(8,2),
            flat_model VARCHAR(50),
            lease_commence_date INTEGER,
            remaining_lease VARCHAR(50),
            resale_price DECIMAL(12,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_town ON resale_transactions(town);
        CREATE INDEX IF NOT EXISTS idx_flat_type ON resale_transactions(flat_type);
        CREATE INDEX IF NOT EXISTS idx_month ON resale_transactions(month);
        CREATE INDEX IF NOT EXISTS idx_resale_price ON resale_transactions(resale_price);
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(schema_sql))
            conn.commit()
            
        logger.info("Database schema created successfully")
    
    def load_data_to_database(self, df):
        """
        Load processed data to PostgreSQL database
        """
        if df.empty:
            logger.warning("No data to load")
            return
            
        try:
            df.to_sql('resale_transactions', self.engine, if_exists='append', index=False)
            logger.info(f"Successfully loaded {len(df)} records to database")
        except Exception as e:
            logger.error(f"Error loading data to database: {e}")
    
    def get_data_summary(self):
        """
        Get summary statistics of the data
        """
        query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT town) as unique_towns,
            COUNT(DISTINCT flat_type) as unique_flat_types,
            MIN(resale_price) as min_price,
            MAX(resale_price) as max_price,
            AVG(resale_price) as avg_price,
            MIN(month) as earliest_month,
            MAX(month) as latest_month
        FROM resale_transactions
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            summary = result.fetchone()
            
        return summary
    
    def run_full_ingestion(self, start_date=None, end_date=None):
        """
        Run complete data ingestion pipeline
        """
        logger.info("Starting full HDB data ingestion pipeline")
        
        # Create schema
        self.create_database_schema()
        
        # Fetch data
        raw_data = self.fetch_hdb_data(start_date, end_date)
        
        # Process data
        processed_data = self.process_raw_data(raw_data)
        
        # Load to database
        if not processed_data.empty:
            self.load_data_to_database(processed_data)
            
            # Get summary
            summary = self.get_data_summary()
            logger.info(f"Data ingestion complete. Summary: {summary}")
        else:
            logger.warning("No data processed")

if __name__ == "__main__":
    # Initialize and run ingestion
    ingestion = HDBDataIngestion()
    ingestion.run_full_ingestion()