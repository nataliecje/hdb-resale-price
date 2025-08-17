import pandas as pd
import numpy as np
import joblib
import pickle
import logging
import time
import json
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitoring:
    def __init__(self):
        # Database configuration
        DB_USER = 'postgres'
        DB_PASSWORD = 'mysecretpassword'
        DB_HOST = 'localhost'
        DB_PORT = '5432'
        DB_NAME = 'hdb'
        
        self.engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
        self.monitoring_data = {}
        
    def check_model_performance(self) -> Dict:
        """
        Monitor model performance metrics
        """
        try:
            # Load model and results
            if not os.path.exists('model_results.pkl'):
                return {"error": "Model results not found"}
            
            with open('model_results.pkl', 'rb') as f:
                results = pickle.load(f)
            
            # Calculate performance metrics
            performance_metrics = {}
            for model_name, model_data in results.items():
                performance_metrics[model_name] = {
                    'mae': model_data['mae'],
                    'rmse': model_data['rmse'],
                    'r2': model_data['r2'],
                    'cv_mean': model_data['cv_mean'],
                    'cv_std': model_data['cv_std']
                }
            
            # Check for performance degradation
            alerts = self._check_performance_alerts(performance_metrics)
            
            return {
                'performance_metrics': performance_metrics,
                'alerts': alerts,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking model performance: {e}")
            return {"error": str(e)}
    
    def _check_performance_alerts(self, metrics: Dict) -> List[str]:
        """
        Check for performance degradation alerts
        """
        alerts = []
        
        # Define thresholds
        thresholds = {
            'r2': 0.7,  # Minimum R² score
            'mae': 50000,  # Maximum MAE in SGD
            'cv_std': 0.1  # Maximum CV standard deviation
        }
        
        for model_name, model_metrics in metrics.items():
            if model_metrics['r2'] < thresholds['r2']:
                alerts.append(f"Low R² score for {model_name}: {model_metrics['r2']:.3f}")
            
            if model_metrics['mae'] > thresholds['mae']:
                alerts.append(f"High MAE for {model_name}: ${model_metrics['mae']:,.0f}")
            
            if model_metrics['cv_std'] > thresholds['cv_std']:
                alerts.append(f"High CV variability for {model_name}: {model_metrics['cv_std']:.3f}")
        
        return alerts
    
    def check_data_quality(self) -> Dict:
        """
        Monitor data quality metrics
        """
        try:
            # Get data quality metrics
            query = """
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT town) as unique_towns,
                COUNT(DISTINCT flat_type) as unique_flat_types,
                COUNT(CASE WHEN resale_price IS NULL THEN 1 END) as null_prices,
                COUNT(CASE WHEN floor_area_sqm IS NULL THEN 1 END) as null_area,
                COUNT(CASE WHEN remaining_lease IS NULL THEN 1 END) as null_lease,
                AVG(resale_price) as avg_price,
                STDDEV(resale_price) as price_std,
                MIN(resale_price) as min_price,
                MAX(resale_price) as max_price,
                MIN(TO_DATE(month, 'YYYY-MM')) as earliest_date,
                MAX(TO_DATE(month, 'YYYY-MM')) as latest_date
            FROM resale_transactions
            """
            
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                data_quality = result.fetchone()
            
            # Calculate quality metrics
            quality_metrics = {
                'total_records': data_quality[0],
                'unique_towns': data_quality[1],
                'unique_flat_types': data_quality[2],
                'null_prices_pct': (data_quality[3] / data_quality[0]) * 100 if data_quality[0] > 0 else 0,
                'null_area_pct': (data_quality[4] / data_quality[0]) * 100 if data_quality[0] > 0 else 0,
                'null_lease_pct': (data_quality[5] / data_quality[0]) * 100 if data_quality[0] > 0 else 0,
                'avg_price': data_quality[6],
                'price_std': data_quality[7],
                'price_range': data_quality[9] - data_quality[8],
                'data_span_days': (data_quality[11] - data_quality[10]).days if data_quality[10] and data_quality[11] else 0
            }
            
            # Check for data quality issues
            alerts = self._check_data_quality_alerts(quality_metrics)
            
            return {
                'quality_metrics': quality_metrics,
                'alerts': alerts,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking data quality: {e}")
            return {"error": str(e)}
    
    def _check_data_quality_alerts(self, metrics: Dict) -> List[str]:
        """
        Check for data quality issues
        """
        alerts = []
        
        # Define thresholds
        thresholds = {
            'null_prices_pct': 5.0,  # Maximum 5% null prices
            'null_area_pct': 10.0,   # Maximum 10% null area
            'null_lease_pct': 15.0,  # Maximum 15% null lease
            'min_records': 10000,    # Minimum records
            'min_towns': 20,         # Minimum towns
            'price_std_threshold': 200000  # Maximum price standard deviation
        }
        
        if metrics['null_prices_pct'] > thresholds['null_prices_pct']:
            alerts.append(f"High null price percentage: {metrics['null_prices_pct']:.1f}%")
        
        if metrics['null_area_pct'] > thresholds['null_area_pct']:
            alerts.append(f"High null area percentage: {metrics['null_area_pct']:.1f}%")
        
        if metrics['null_lease_pct'] > thresholds['null_lease_pct']:
            alerts.append(f"High null lease percentage: {metrics['null_lease_pct']:.1f}%")
        
        if metrics['total_records'] < thresholds['min_records']:
            alerts.append(f"Low record count: {metrics['total_records']:,}")
        
        if metrics['unique_towns'] < thresholds['min_towns']:
            alerts.append(f"Low town diversity: {metrics['unique_towns']}")
        
        if metrics['price_std'] > thresholds['price_std_threshold']:
            alerts.append(f"High price variability: ${metrics['price_std']:,.0f}")
        
        return alerts
    
    def check_feature_drift(self) -> Dict:
        """
        Monitor feature distribution drift
        """
        try:
            # Get recent data (last 3 months)
            recent_query = """
            SELECT 
                AVG(resale_price) as recent_avg_price,
                STDDEV(resale_price) as recent_price_std,
                AVG(floor_area_sqm) as recent_avg_area,
                COUNT(*) as recent_count
            FROM resale_transactions 
            WHERE TO_DATE(month, 'YYYY-MM') >= CURRENT_DATE - INTERVAL '3 months'
            """
            
            # Get historical data (3-6 months ago)
            historical_query = """
            SELECT 
                AVG(resale_price) as historical_avg_price,
                STDDEV(resale_price) as historical_price_std,
                AVG(floor_area_sqm) as historical_avg_area,
                COUNT(*) as historical_count
            FROM resale_transactions 
            WHERE TO_DATE(month, 'YYYY-MM') BETWEEN CURRENT_DATE - INTERVAL '6 months' 
                AND CURRENT_DATE - INTERVAL '3 months'
            """
            
            with self.engine.connect() as conn:
                recent_data = conn.execute(text(recent_query)).fetchone()
                historical_data = conn.execute(text(historical_query)).fetchone()
            
            # Calculate drift metrics
            drift_metrics = {
                'price_drift_pct': ((recent_data[0] - historical_data[0]) / historical_data[0]) * 100 if historical_data[0] else 0,
                'area_drift_pct': ((recent_data[2] - historical_data[2]) / historical_data[2]) * 100 if historical_data[2] else 0,
                'volume_change_pct': ((recent_data[3] - historical_data[3]) / historical_data[3]) * 100 if historical_data[3] else 0,
                'recent_avg_price': recent_data[0],
                'historical_avg_price': historical_data[0],
                'recent_count': recent_data[3],
                'historical_count': historical_data[3]
            }
            
            # Check for significant drift
            alerts = self._check_drift_alerts(drift_metrics)
            
            return {
                'drift_metrics': drift_metrics,
                'alerts': alerts,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error checking feature drift: {e}")
            return {"error": str(e)}
    
    def _check_drift_alerts(self, metrics: Dict) -> List[str]:
        """
        Check for significant feature drift
        """
        alerts = []
        
        # Define drift thresholds
        thresholds = {
            'price_drift_pct': 10.0,  # 10% price change
            'area_drift_pct': 5.0,    # 5% area change
            'volume_change_pct': 50.0  # 50% volume change
        }
        
        if abs(metrics['price_drift_pct']) > thresholds['price_drift_pct']:
            direction = "increase" if metrics['price_drift_pct'] > 0 else "decrease"
            alerts.append(f"Significant price {direction}: {metrics['price_drift_pct']:.1f}%")
        
        if abs(metrics['area_drift_pct']) > thresholds['area_drift_pct']:
            direction = "increase" if metrics['area_drift_pct'] > 0 else "decrease"
            alerts.append(f"Significant area {direction}: {metrics['area_drift_pct']:.1f}%")
        
        if abs(metrics['volume_change_pct']) > thresholds['volume_change_pct']:
            direction = "increase" if metrics['volume_change_pct'] > 0 else "decrease"
            alerts.append(f"Significant volume {direction}: {metrics['volume_change_pct']:.1f}%")
        
        return alerts
    
    def generate_monitoring_report(self) -> Dict:
        """
        Generate comprehensive monitoring report
        """
        try:
            # Collect all monitoring data
            performance = self.check_model_performance()
            data_quality = self.check_data_quality()
            feature_drift = self.check_feature_drift()
            
            # Combine all alerts
            all_alerts = []
            if 'alerts' in performance:
                all_alerts.extend(performance['alerts'])
            if 'alerts' in data_quality:
                all_alerts.extend(data_quality['alerts'])
            if 'alerts' in feature_drift:
                all_alerts.extend(feature_drift['alerts'])
            
            # Determine overall system health
            system_health = "healthy" if len(all_alerts) == 0 else "warning" if len(all_alerts) <= 3 else "critical"
            
            report = {
                'system_health': system_health,
                'total_alerts': len(all_alerts),
                'alerts': all_alerts,
                'performance': performance,
                'data_quality': data_quality,
                'feature_drift': feature_drift,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save report
            with open('monitoring_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating monitoring report: {e}")
            return {"error": str(e)}
    
    def create_monitoring_dashboard(self) -> str:
        """
        Create monitoring dashboard plots
        """
        try:
            # Load monitoring data
            if os.path.exists('monitoring_report.json'):
                with open('monitoring_report.json', 'r') as f:
                    report = json.load(f)
            else:
                report = self.generate_monitoring_report()
            
            # Create dashboard plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('HDB BTO System Monitoring Dashboard', fontsize=16)
            
            # 1. System Health Overview
            health_colors = {'healthy': 'green', 'warning': 'orange', 'critical': 'red'}
            health = report.get('system_health', 'unknown')
            axes[0, 0].text(0.5, 0.5, f"System Health: {health.upper()}", 
                           ha='center', va='center', fontsize=20, 
                           color=health_colors.get(health, 'black'))
            axes[0, 0].set_title('System Health Status')
            axes[0, 0].axis('off')
            
            # 2. Alerts Summary
            alerts = report.get('alerts', [])
            alert_categories = {}
            for alert in alerts:
                category = alert.split(':')[0] if ':' in alert else 'Other'
                alert_categories[category] = alert_categories.get(category, 0) + 1
            
            if alert_categories:
                axes[0, 1].pie(alert_categories.values(), labels=alert_categories.keys(), autopct='%1.1f%%')
                axes[0, 1].set_title('Alert Categories')
            else:
                axes[0, 1].text(0.5, 0.5, 'No Alerts', ha='center', va='center', fontsize=16)
                axes[0, 1].set_title('Alert Categories')
            
            # 3. Performance Metrics (if available)
            if 'performance' in report and 'performance_metrics' in report['performance']:
                perf_metrics = report['performance']['performance_metrics']
                models = list(perf_metrics.keys())
                r2_scores = [perf_metrics[model]['r2'] for model in models]
                
                axes[1, 0].bar(models, r2_scores)
                axes[1, 0].set_title('Model R² Scores')
                axes[1, 0].set_ylabel('R² Score')
                axes[1, 0].tick_params(axis='x', rotation=45)
            else:
                axes[1, 0].text(0.5, 0.5, 'No Performance Data', ha='center', va='center', fontsize=16)
                axes[1, 0].set_title('Model Performance')
            
            # 4. Data Quality Metrics (if available)
            if 'data_quality' in report and 'quality_metrics' in report['data_quality']:
                quality_metrics = report['data_quality']['quality_metrics']
                metrics_names = ['Null Prices', 'Null Area', 'Null Lease']
                null_percentages = [
                    quality_metrics.get('null_prices_pct', 0),
                    quality_metrics.get('null_area_pct', 0),
                    quality_metrics.get('null_lease_pct', 0)
                ]
                
                axes[1, 1].bar(metrics_names, null_percentages)
                axes[1, 1].set_title('Data Quality - Null Percentages')
                axes[1, 1].set_ylabel('Percentage (%)')
                axes[1, 1].tick_params(axis='x', rotation=45)
            else:
                axes[1, 1].text(0.5, 0.5, 'No Quality Data', ha='center', va='center', fontsize=16)
                axes[1, 1].set_title('Data Quality')
            
            plt.tight_layout()
            plt.savefig('monitoring_dashboard.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return "monitoring_dashboard.png"
            
        except Exception as e:
            logger.error(f"Error creating monitoring dashboard: {e}")
            return None

# Example usage
if __name__ == "__main__":
    monitor = ModelMonitoring()
    
    # Generate monitoring report
    report = monitor.generate_monitoring_report()
    print("Monitoring Report:")
    print(json.dumps(report, indent=2))
    
    # Create dashboard
    dashboard_file = monitor.create_monitoring_dashboard()
    if dashboard_file:
        print(f"Dashboard saved as: {dashboard_file}")
