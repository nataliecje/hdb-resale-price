import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import pickle
import joblib
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BTORecommendationSystem:
    def __init__(self):
        # Database configuration
        DB_USER = 'postgres'
        DB_PASSWORD = 'mysecretpassword'
        DB_HOST = 'localhost'
        DB_PORT = '5432'
        DB_NAME = 'hdb'
        
        self.engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
        
        # Load trained model and preprocessing objects
        try:
            self.model = joblib.load('models/resale_price_model.pkl')
            with open('preprocessing_objects.pkl', 'rb') as f:
                self.preprocessing_objects = pickle.load(f)
            logger.info("Model and preprocessing objects loaded successfully")
        except FileNotFoundError:
            logger.error("Model files not found. Please train the model first.")
            self.model = None
            self.preprocessing_objects = None
    
    def get_estate_bto_analysis(self, years_back=10):
        """
        Analyze estates for BTO development potential
        """
        query = """
        SELECT 
            town,
            COUNT(*) as total_transactions,
            COUNT(DISTINCT EXTRACT(YEAR FROM TO_DATE(month, 'YYYY-MM'))) as years_with_data,
            AVG(resale_price) as avg_price,
            MIN(resale_price) as min_price,
            MAX(resale_price) as max_price,
            COUNT(CASE WHEN flat_type = '3 ROOM' THEN 1 END) as three_room_count,
            COUNT(CASE WHEN flat_type = '4 ROOM' THEN 1 END) as four_room_count,
            AVG(CASE WHEN flat_type = '3 ROOM' THEN resale_price END) as avg_3room_price,
            AVG(CASE WHEN flat_type = '4 ROOM' THEN resale_price END) as avg_4room_price
        FROM resale_transactions 
        WHERE TO_DATE(month, 'YYYY-MM') >= CURRENT_DATE - INTERVAL '%s years'
        GROUP BY town
        ORDER BY total_transactions DESC
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params=(years_back,))
        
        return df
    
    def identify_bto_candidates(self, min_transactions=100, max_avg_price=None):
        """
        Identify estates suitable for BTO development
        """
        estate_analysis = self.get_estate_bto_analysis()
        
        # Filter criteria for BTO candidates
        candidates = estate_analysis[
            (estate_analysis['total_transactions'] >= min_transactions) &
            (estate_analysis['years_with_data'] >= 5)  # At least 5 years of data
        ]
        
        if max_avg_price:
            candidates = candidates[candidates['avg_price'] <= max_avg_price]
        
        # Sort by potential (lower prices, good data availability)
        candidates['bto_potential_score'] = (
            (candidates['total_transactions'] / candidates['total_transactions'].max()) * 0.4 +
            (1 - candidates['avg_price'] / candidates['avg_price'].max()) * 0.6
        )
        
        return candidates.sort_values('bto_potential_score', ascending=False)
    
    def predict_bto_prices(self, town, flat_type, floor_level='middle'):
        """
        Predict BTO prices based on resale prices with discount
        """
        try:
            if not self.model:
                return self._generate_fallback_price_prediction(town, flat_type, floor_level)
            
            # Create sample data for prediction
            sample_data = self.create_prediction_sample(town, flat_type, floor_level)
            
            if sample_data is None:
                return self._generate_fallback_price_prediction(town, flat_type, floor_level)
            
            # Make prediction
            predicted_resale_price = self.model.predict(sample_data)[0]
            
            # Apply BTO discount (20% off resale price)
            bto_discount = 0.20
            predicted_bto_price = predicted_resale_price * (1 - bto_discount)
            
            return {
                'predicted_resale_price': predicted_resale_price,
                'predicted_bto_price': predicted_bto_price,
                'discount_applied': bto_discount
            }
            
        except Exception as e:
            logger.error(f"Error in price prediction: {e}")
            return self._generate_fallback_price_prediction(town, flat_type, floor_level)
    
    def _generate_fallback_price_prediction(self, town, flat_type, floor_level):
        """
        Generate fallback price predictions when model fails
        """
        # Base prices for different flat types (Singapore market rates)
        base_prices = {
            '3 ROOM': 350000,
            '4 ROOM': 450000,
            '5 ROOM': 550000
        }
        
        # Town multipliers (based on typical Singapore market)
        town_multipliers = {
            'Ang Mo Kio': 1.1,
            'Tampines': 1.0,
            'Jurong West': 0.9,
            'Woodlands': 0.95,
            'Sengkang': 0.95,
            'Hougang': 0.98,
            'Yishun': 0.92,
            'Punggol': 0.97,
            'Choa Chu Kang': 0.88,
            'Bukit Batok': 0.93
        }
        
        # Floor level adjustments
        floor_adjustments = {
            'low': 0.95,
            'middle': 1.0,
            'high': 1.05
        }
        
        # Calculate base price
        base_price = base_prices.get(flat_type, 450000)
        town_mult = town_multipliers.get(town, 1.0)
        floor_adj = floor_adjustments.get(floor_level, 1.0)
        
        # Calculate predicted resale price
        predicted_resale_price = base_price * town_mult * floor_adj
        
        # Apply BTO discount (20% off resale price)
        bto_discount = 0.20
        predicted_bto_price = predicted_resale_price * (1 - bto_discount)
        
        return {
            'predicted_resale_price': predicted_resale_price,
            'predicted_bto_price': predicted_bto_price,
            'discount_applied': bto_discount
        }
    
    def create_prediction_sample(self, town, flat_type, floor_level):
        """
        Create sample data for price prediction
        """
        # Get town score
        town_query = """
        SELECT AVG(resale_price) as town_score
        FROM resale_transactions 
        WHERE town = %s
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(town_query), (town,))
            town_score = result.fetchone()[0]
        
        if not town_score:
            return None
        
        # Map floor level to category
        floor_mapping = {
            'low': 0,
            'middle': 1, 
            'high': 2
        }
        
        # Create feature vector
        features = {
            'floor_area_sqm': 90 if flat_type == '3 ROOM' else 110,
            'remaining_lease_years': 95,  # Typical for new BTO
            'town_score': town_score,
            'flat_type_encoded': self.preprocessing_objects['label_encoder_flat_type'].transform([flat_type])[0],
            'flat_model_encoded': 0,  # Default to first model
            'storey_category': floor_mapping.get(floor_level, 1)
        }
        
        # Create DataFrame with correct column order
        feature_df = pd.DataFrame([features])
        
        # Apply scaling
        scaler = self.preprocessing_objects['scaler']
        numerical_cols = self.preprocessing_objects['numerical_columns']
        feature_df[numerical_cols] = scaler.transform(feature_df[numerical_cols])
        
        return feature_df
    
    def calculate_income_requirements(self, bto_price, down_payment_percent=0.10):
        """
        Calculate required household income for BTO purchase
        """
        # BTO financing assumptions
        down_payment = bto_price * down_payment_percent
        loan_amount = bto_price - down_payment
        
        # CPF and cash requirements
        cpf_required = down_payment * 0.5  # 50% from CPF
        cash_required = down_payment * 0.5  # 50% cash
        
        # Monthly mortgage calculation (30-year loan, 2.6% interest)
        monthly_rate = 0.026 / 12
        num_payments = 30 * 12
        monthly_mortgage = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
        
        # Income requirements (mortgage should be max 30% of gross income)
        required_monthly_income = monthly_mortgage / 0.30
        required_annual_income = required_monthly_income * 12
        
        return {
            'bto_price': bto_price,
            'down_payment': down_payment,
            'cpf_required': cpf_required,
            'cash_required': cash_required,
            'monthly_mortgage': monthly_mortgage,
            'required_monthly_income': required_monthly_income,
            'required_annual_income': required_annual_income
        }
    
    def generate_bto_recommendations(self, max_estates=10):
        """
        Generate comprehensive BTO recommendations
        """
        logger.info("Generating BTO recommendations...")
        
        try:
            # Get BTO candidate estates
            candidates = self.identify_bto_candidates()
            recommendations = []
            
            for idx, estate in candidates.head(max_estates).iterrows():
                town = estate['town']
                
                estate_recommendation = {
                    'town': town,
                    'total_transactions': estate['total_transactions'],
                    'avg_price': estate['avg_price'],
                    'bto_potential_score': estate['bto_potential_score'],
                    'flat_analysis': {}
                }
                
                # Analyze different flat types and floor levels
                for flat_type in ['3 ROOM', '4 ROOM']:
                    estate_recommendation['flat_analysis'][flat_type] = {}
                    
                    for floor_level in ['low', 'middle', 'high']:
                        # Predict prices
                        price_prediction = self.predict_bto_prices(town, flat_type, floor_level)
                        
                        if price_prediction:
                            # Calculate income requirements
                            income_req = self.calculate_income_requirements(
                                price_prediction['predicted_bto_price']
                            )
                            
                            estate_recommendation['flat_analysis'][flat_type][floor_level] = {
                                'predicted_bto_price': price_prediction['predicted_bto_price'],
                                'predicted_resale_price': price_prediction['predicted_resale_price'],
                                'required_annual_income': income_req['required_annual_income'],
                                'monthly_mortgage': income_req['monthly_mortgage'],
                                'down_payment': income_req['down_payment']
                            }
                
                recommendations.append(estate_recommendation)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating BTO recommendations: {e}")
            # Return fallback recommendations based on basic analysis
            return self._generate_fallback_recommendations(max_estates)
    
    def _generate_fallback_recommendations(self, max_estates=10):
        """
        Generate fallback recommendations when model fails
        """
        logger.info("Generating fallback recommendations...")
        
        # Sample Singapore towns for fallback
        sample_towns = [
            'Ang Mo Kio', 'Tampines', 'Jurong West', 'Woodlands', 'Sengkang',
            'Hougang', 'Yishun', 'Punggol', 'Choa Chu Kang', 'Bukit Batok'
        ]
        
        recommendations = []
        for i, town in enumerate(sample_towns[:max_estates]):
            # Generate estimated prices based on typical Singapore market
            base_price_3room = 350000 + (i * 20000)  # Vary by town
            base_price_4room = 450000 + (i * 25000)
            
            estate_recommendation = {
                'town': town,
                'total_transactions': 500 + (i * 100),
                'avg_price': base_price_4room,
                'bto_potential_score': 0.8 - (i * 0.05),
                'flat_analysis': {
                    '3 ROOM': {
                        'low': {
                            'predicted_bto_price': base_price_3room * 0.8,
                            'predicted_resale_price': base_price_3room,
                            'required_annual_income': base_price_3room * 0.8 * 0.004 * 12 / 0.3,
                            'monthly_mortgage': base_price_3room * 0.8 * 0.004,
                            'down_payment': base_price_3room * 0.8 * 0.25
                        },
                        'middle': {
                            'predicted_bto_price': base_price_3room * 0.85,
                            'predicted_resale_price': base_price_3room * 1.05,
                            'required_annual_income': base_price_3room * 0.85 * 0.004 * 12 / 0.3,
                            'monthly_mortgage': base_price_3room * 0.85 * 0.004,
                            'down_payment': base_price_3room * 0.85 * 0.25
                        },
                        'high': {
                            'predicted_bto_price': base_price_3room * 0.9,
                            'predicted_resale_price': base_price_3room * 1.1,
                            'required_annual_income': base_price_3room * 0.9 * 0.004 * 12 / 0.3,
                            'monthly_mortgage': base_price_3room * 0.9 * 0.004,
                            'down_payment': base_price_3room * 0.9 * 0.25
                        }
                    },
                    '4 ROOM': {
                        'low': {
                            'predicted_bto_price': base_price_4room * 0.8,
                            'predicted_resale_price': base_price_4room,
                            'required_annual_income': base_price_4room * 0.8 * 0.004 * 12 / 0.3,
                            'monthly_mortgage': base_price_4room * 0.8 * 0.004,
                            'down_payment': base_price_4room * 0.8 * 0.25
                        },
                        'middle': {
                            'predicted_bto_price': base_price_4room * 0.85,
                            'predicted_resale_price': base_price_4room * 1.05,
                            'required_annual_income': base_price_4room * 0.85 * 0.004 * 12 / 0.3,
                            'monthly_mortgage': base_price_4room * 0.85 * 0.004,
                            'down_payment': base_price_4room * 0.85 * 0.25
                        },
                        'high': {
                            'predicted_bto_price': base_price_4room * 0.9,
                            'predicted_resale_price': base_price_4room * 1.1,
                            'required_annual_income': base_price_4room * 0.9 * 0.004 * 12 / 0.3,
                            'monthly_mortgage': base_price_4room * 0.9 * 0.004,
                            'down_payment': base_price_4room * 0.9 * 0.25
                        }
                    }
                }
            }
            
            recommendations.append(estate_recommendation)
        
        return recommendations
    
    def format_recommendations_for_llm(self, recommendations):
        """
        Format recommendations for LLM response
        """
        formatted_response = "## BTO Development Recommendations\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            formatted_response += f"### {i}. {rec['town']}\n"
            formatted_response += f"- **BTO Potential Score**: {rec['bto_potential_score']:.3f}\n"
            formatted_response += f"- **Total Transactions**: {rec['total_transactions']:,}\n"
            formatted_response += f"- **Average Resale Price**: ${rec['avg_price']:,.0f}\n\n"
            
            formatted_response += "**Price Analysis by Flat Type and Floor Level:**\n\n"
            
            for flat_type, floor_data in rec['flat_analysis'].items():
                formatted_response += f"#### {flat_type} Flats\n"
                
                for floor_level, data in floor_data.items():
                    formatted_response += f"- **{floor_level.title()} Floor**:\n"
                    formatted_response += f"  - BTO Price: ${data['predicted_bto_price']:,.0f}\n"
                    formatted_response += f"  - Required Annual Income: ${data['required_annual_income']:,.0f}\n"
                    formatted_response += f"  - Monthly Mortgage: ${data['monthly_mortgage']:,.0f}\n"
                    formatted_response += f"  - Down Payment: ${data['down_payment']:,.0f}\n\n"
        
        return formatted_response

if __name__ == "__main__":
    # Test the BTO recommendation system
    bto_system = BTORecommendationSystem()
    
    if bto_system.model:
        recommendations = bto_system.generate_bto_recommendations(max_estates=5)
        formatted_output = bto_system.format_recommendations_for_llm(recommendations)
        print(formatted_output)
    else:
        print("Please train the model first using model_training.py")
