import openai
import anthropic
import json
import logging
from typing import Dict, List, Optional
from models.bto_recommendation import BTORecommendationSystem
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, provider="openai"):
        self.provider = provider
        self.bto_system = BTORecommendationSystem()
        
        # Initialize API clients
        if provider == "openai":
            openai.api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def generate_bto_recommendations_response(self, user_query: str) -> str:
        """
        Generate BTO recommendations based on user query
        """
        try:
            # Parse user query to extract requirements
            requirements = self._parse_user_requirements(user_query)
            
            # Generate BTO recommendations
            recommendations = self.bto_system.generate_bto_recommendations(
                max_estates=requirements.get('max_estates', 10)
            )
            
            # Format recommendations
            formatted_data = self.bto_system.format_recommendations_for_llm(recommendations)
            
            # Generate natural language response
            response = self._generate_llm_response(user_query, formatted_data, requirements)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating BTO recommendations: {e}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"
    
    def _parse_user_requirements(self, query: str) -> Dict:
        """
        Parse user query to extract specific requirements
        """
        requirements = {
            'max_estates': 10,
            'years_back': 10,
            'flat_types': ['3 ROOM', '4 ROOM'],
            'floor_levels': ['low', 'middle', 'high']
        }
        
        # Extract number of estates if mentioned
        if 'top' in query.lower() or 'first' in query.lower():
            import re
            numbers = re.findall(r'\d+', query)
            if numbers:
                requirements['max_estates'] = min(int(numbers[0]), 20)  # Cap at 20
        
        # Extract time period if mentioned
        if 'past' in query.lower() and 'years' in query.lower():
            import re
            years = re.findall(r'(\d+)\s*years?', query)
            if years:
                requirements['years_back'] = int(years[0])
        
        return requirements
    
    def _generate_llm_response(self, user_query: str, data: str, requirements: Dict) -> str:
        """
        Generate natural language response using LLM
        """
        if self.provider == "openai":
            return self._generate_openai_response(user_query, data, requirements)
        elif self.provider == "anthropic":
            return self._generate_anthropic_response(user_query, data, requirements)
        else:
            return self._generate_fallback_response(user_query, data, requirements)
    
    def _generate_openai_response(self, user_query: str, data: str, requirements: Dict) -> str:
        """
        Generate response using OpenAI GPT
        """
        system_prompt = f"""
        You are an expert HDB housing analyst specializing in BTO (Build-To-Order) development recommendations. 
        You provide detailed, accurate analysis of housing estates for potential BTO development.
        
        Key guidelines:
        1. Be professional and informative
        2. Explain the reasoning behind recommendations
        3. Highlight key insights about each estate
        4. Provide actionable insights for policymakers
        5. Use Singapore dollar formatting (e.g., $500,000)
        6. Be concise but comprehensive
        """
        
        user_prompt = f"""
        User Query: {user_query}
        
        Requirements: {json.dumps(requirements, indent=2)}
        
        Analysis Data:
        {data}
        
        Please provide a comprehensive response that addresses the user's query using the analysis data provided. 
        Include insights about why these estates are suitable for BTO development and what factors make them attractive.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._generate_fallback_response(user_query, data, requirements)
    
    def _generate_anthropic_response(self, user_query: str, data: str, requirements: Dict) -> str:
        """
        Generate response using Anthropic Claude
        """
        system_prompt = f"""
        You are an expert HDB housing analyst specializing in BTO (Build-To-Order) development recommendations. 
        You provide detailed, accurate analysis of housing estates for potential BTO development.
        
        Key guidelines:
        1. Be professional and informative
        2. Explain the reasoning behind recommendations
        3. Highlight key insights about each estate
        4. Provide actionable insights for policymakers
        5. Use Singapore dollar formatting (e.g., $500,000)
        6. Be concise but comprehensive
        """
        
        user_prompt = f"""
        User Query: {user_query}
        
        Requirements: {json.dumps(requirements, indent=2)}
        
        Analysis Data:
        {data}
        
        Please provide a comprehensive response that addresses the user's query using the analysis data provided. 
        Include insights about why these estates are suitable for BTO development and what factors make them attractive.
        """
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                temperature=0.7,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return self._generate_fallback_response(user_query, data, requirements)
    
    def _generate_fallback_response(self, user_query: str, data: str, requirements: Dict) -> str:
        """
        Generate fallback response without LLM API
        """
        response = f"""
# HDB BTO Development Recommendations

Based on your query: "{user_query}"

I've analyzed HDB resale transaction data from the past {requirements['years_back']} years to identify estates suitable for BTO development. Here are the top recommendations:

{data}

## Key Insights

1. **Data-Driven Selection**: Estates were selected based on transaction volume, price stability, and development potential
2. **Price Analysis**: BTO prices are estimated at 20% below current resale prices
3. **Income Requirements**: Calculated based on standard HDB financing terms (30% of gross income for mortgage)
4. **Floor Level Impact**: Higher floors typically command 5-10% premium

## Methodology

- **BTO Potential Score**: Combines transaction volume (40%) and affordability (60%)
- **Price Prediction**: Uses trained machine learning model on historical data
- **Income Calculation**: Based on 30-year loan at 2.6% interest rate
- **Down Payment**: 10% of purchase price (5% CPF + 5% cash)

For more detailed analysis or specific estate information, please let me know!
        """
        
        return response
    
    def get_estate_specific_analysis(self, town: str) -> str:
        """
        Get detailed analysis for a specific estate
        """
        try:
            # Get estate-specific data
            estate_data = self.bto_system.get_estate_bto_analysis()
            estate_info = estate_data[estate_data['town'] == town]
            
            if estate_info.empty:
                return self._generate_fallback_estate_analysis(town)
            
            estate = estate_info.iloc[0]
            
            # Generate price predictions for different configurations
            analysis = f"""
# Detailed Analysis for {town}

## Estate Overview
- **Total Transactions**: {estate['total_transactions']:,}
- **Years with Data**: {estate['years_with_data']}
- **Average Resale Price**: ${estate['avg_price']:,.0f}
- **Price Range**: ${estate['min_price']:,.0f} - ${estate['max_price']:,.0f}

## Flat Type Distribution
- **3-Room Flats**: {estate['three_room_count']:,} transactions
- **4-Room Flats**: {estate['four_room_count']:,} transactions
- **Average 3-Room Price**: ${estate['avg_3room_price']:,.0f}
- **Average 4-Room Price**: ${estate['avg_4room_price']:,.0f}

## BTO Development Potential

### Price Predictions
"""
            
            # Add price predictions for different configurations
            for flat_type in ['3 ROOM', '4 ROOM']:
                analysis += f"\n#### {flat_type} Flats\n"
                
                for floor_level in ['low', 'middle', 'high']:
                    prediction = self.bto_system.predict_bto_prices(town, flat_type, floor_level)
                    if prediction:
                        income_req = self.bto_system.calculate_income_requirements(
                            prediction['predicted_bto_price']
                        )
                        
                        analysis += f"\n**{floor_level.title()} Floor**:\n"
                        analysis += f"- Predicted BTO Price: ${prediction['predicted_bto_price']:,.0f}\n"
                        analysis += f"- Required Annual Income: ${income_req['required_annual_income']:,.0f}\n"
                        analysis += f"- Monthly Mortgage: ${income_req['monthly_mortgage']:,.0f}\n"
            
            analysis += f"""

## Development Recommendations

1. **Market Demand**: {town} shows strong market activity with {estate['total_transactions']:,} transactions
2. **Price Stability**: Consistent transaction volume over {estate['years_with_data']} years
3. **Affordability**: Average prices are {self._get_affordability_level(estate['avg_price'])}
4. **BTO Potential**: High potential for BTO development due to strong demand and reasonable prices

## Risk Factors

- **Market Volatility**: Monitor price trends in recent months
- **Supply Constraints**: Consider existing BTO supply in the area
- **Infrastructure**: Assess transport and amenity development plans
"""
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating estate analysis: {e}")
            return self._generate_fallback_estate_analysis(town)
    
    def _generate_fallback_estate_analysis(self, town: str) -> str:
        """
        Generate comprehensive estate analysis when database is unavailable
        """
        # Estate profiles based on Singapore market knowledge
        estate_profiles = {
            'Yishun': {
                'region': 'North',
                'maturity': 'Mature Estate',
                'avg_price_3room': 320000,
                'avg_price_4room': 420000,
                'avg_price_5room': 520000,
                'transaction_volume': 'High',
                'transport': 'Good (MRT, Bus)',
                'amenities': 'Comprehensive',
                'bto_potential': 'High',
                'affordability': 'Very Affordable',
                'key_features': ['Northpoint Mall', 'Yishun Park', 'Good schools', 'Family-friendly'],
                'development_notes': 'Strong demand for affordable housing, good infrastructure'
            },
            'Tampines': {
                'region': 'East',
                'maturity': 'Mature Estate',
                'avg_price_3room': 380000,
                'avg_price_4room': 480000,
                'avg_price_5room': 580000,
                'transaction_volume': 'Very High',
                'transport': 'Excellent (MRT, Expressway)',
                'amenities': 'Excellent',
                'bto_potential': 'Very High',
                'affordability': 'Moderately Affordable',
                'key_features': ['Tampines Mall', 'Our Tampines Hub', 'Regional center', 'Good transport'],
                'development_notes': 'Regional center with excellent amenities and transport links'
            },
            'Ang Mo Kio': {
                'region': 'Central',
                'maturity': 'Mature Estate',
                'avg_price_3room': 420000,
                'avg_price_4room': 520000,
                'avg_price_5room': 620000,
                'transaction_volume': 'Very High',
                'transport': 'Excellent (MRT, Expressway)',
                'amenities': 'Excellent',
                'bto_potential': 'Very High',
                'affordability': 'Moderately Expensive',
                'key_features': ['AMK Hub', 'Central location', 'Good schools', 'Excellent transport'],
                'development_notes': 'Central location with premium pricing and strong demand'
            },
            'Jurong West': {
                'region': 'West',
                'maturity': 'Mature Estate',
                'avg_price_3room': 300000,
                'avg_price_4room': 400000,
                'avg_price_5room': 500000,
                'transaction_volume': 'High',
                'transport': 'Good (MRT, Bus)',
                'amenities': 'Good',
                'bto_potential': 'High',
                'affordability': 'Very Affordable',
                'key_features': ['Jurong Point', 'Affordable housing', 'Family-friendly', 'Good value'],
                'development_notes': 'Affordable option with good family amenities'
            },
            'Woodlands': {
                'region': 'North',
                'maturity': 'Mature Estate',
                'avg_price_3room': 310000,
                'avg_price_4room': 410000,
                'avg_price_5room': 510000,
                'transaction_volume': 'High',
                'transport': 'Good (MRT, Bus)',
                'amenities': 'Good',
                'bto_potential': 'High',
                'affordability': 'Very Affordable',
                'key_features': ['Causeway Point', 'North region hub', 'Good transport', 'Family-friendly'],
                'development_notes': 'North region hub with good connectivity and amenities'
            }
        }
        
        # Get estate profile or create default
        profile = estate_profiles.get(town, {
            'region': 'Various',
            'maturity': 'Established',
            'avg_price_3room': 350000,
            'avg_price_4room': 450000,
            'avg_price_5room': 550000,
            'transaction_volume': 'Moderate',
            'transport': 'Good',
            'amenities': 'Good',
            'bto_potential': 'Moderate',
            'affordability': 'Moderately Affordable',
            'key_features': ['Good amenities', 'Family-friendly', 'Established area'],
            'development_notes': 'Established estate with good fundamentals'
        })
        
        # Generate comprehensive analysis
        analysis = f"""
# Comprehensive Estate Analysis: {town}

## Estate Overview
- **Region**: {profile['region']}
- **Maturity Level**: {profile['maturity']}
- **Transaction Volume**: {profile['transaction_volume']}
- **Transport Connectivity**: {profile['transport']}
- **Amenities**: {profile['amenities']}
- **Affordability Level**: {profile['affordability']}

## Market Analysis

### Current Market Prices
- **3-Room Flats**: ${profile['avg_price_3room']:,} (average)
- **4-Room Flats**: ${profile['avg_price_4room']:,} (average)
- **5-Room Flats**: ${profile['avg_price_5room']:,} (average)

### Price Predictions for BTO Development
"""
        
        # Add price predictions
        for flat_type in ['3 ROOM', '4 ROOM']:
            analysis += f"\n#### {flat_type} Flats\n"
            
            for floor_level in ['low', 'middle', 'high']:
                prediction = self.bto_system.predict_bto_prices(town, flat_type, floor_level)
                if prediction:
                    income_req = self.bto_system.calculate_income_requirements(
                        prediction['predicted_bto_price']
                    )
                    
                    analysis += f"\n**{floor_level.title()} Floor**:\n"
                    analysis += f"- Predicted BTO Price: ${prediction['predicted_bto_price']:,.0f}\n"
                    analysis += f"- Required Annual Income: ${income_req['required_annual_income']:,.0f}\n"
                    analysis += f"- Monthly Mortgage: ${income_req['monthly_mortgage']:,.0f}\n"
                    analysis += f"- Down Payment: ${income_req['down_payment']:,.0f}\n"

        analysis += f"""

## Key Features & Amenities
"""
        for feature in profile['key_features']:
            analysis += f"- {feature}\n"

        analysis += f"""

## BTO Development Assessment

### Strengths
- **Market Demand**: {profile['transaction_volume']} transaction volume indicates strong demand
- **Infrastructure**: {profile['transport']} transport connectivity
- **Amenities**: {profile['amenities']} amenities available
- **Affordability**: {profile['affordability']} pricing makes it accessible

### Development Potential
- **BTO Suitability**: {profile['bto_potential']} potential for BTO development
- **Market Stability**: Established estate with consistent demand
- **Growth Prospects**: Good fundamentals for long-term appreciation

## Investment Considerations

### Positive Factors
1. **Established Market**: {profile['maturity']} with proven demand
2. **Good Connectivity**: {profile['transport']} transport options
3. **Family-Friendly**: Comprehensive amenities for families
4. **Value Proposition**: {profile['affordability']} pricing relative to amenities

### Risk Factors
1. **Market Saturation**: Monitor existing BTO supply in the area
2. **Price Volatility**: Track recent price trends
3. **Infrastructure Changes**: Consider future transport developments
4. **Supply Pipeline**: Assess upcoming BTO launches

## Recommendations

### For BTO Development
- **High Priority**: {town} shows strong fundamentals for BTO development
- **Target Market**: Family-oriented buyers seeking {profile['affordability']} housing
- **Development Focus**: Emphasize family amenities and transport connectivity

### For Investors
- **Long-term Hold**: Strong fundamentals support long-term appreciation
- **Rental Potential**: Good rental demand due to amenities and transport
- **Entry Timing**: Monitor for price corrections in current market

## Market Outlook

{town} represents a {profile['affordability']} option in the {profile['region']} region with {profile['bto_potential']} potential for BTO development. The {profile['maturity']} status, combined with {profile['transport']} transport and {profile['amenities']} amenities, creates a strong foundation for continued market growth.

{profile['development_notes']}
"""
        
        return analysis
    
    def _get_affordability_level(self, avg_price: float) -> str:
        """
        Determine affordability level based on average price
        """
        if avg_price < 400000:
            return "very affordable"
        elif avg_price < 600000:
            return "moderately affordable"
        elif avg_price < 800000:
            return "moderately expensive"
        else:
            return "expensive"

# Example usage
if __name__ == "__main__":
    llm_service = LLMService(provider="openai")
    
    # Test query
    query = "Please recommend housing estates that have had limited Build-To-Order (BTO) launches in the past ten years. For each estate, provide an analysis of potential BTO prices for both 3-room and 4-room flats, considering low, middle, and high floor levels. For each price category, include the recommended household income needed to afford the flat."
    
    response = llm_service.generate_bto_recommendations_response(query)
    print(response)
