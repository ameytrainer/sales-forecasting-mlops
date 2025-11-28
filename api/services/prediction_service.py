"""
Prediction Service - Business Logic for Predictions

Handles prediction preprocessing, inference, and postprocessing.

Author: Amey Talkatkar
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys

sys.path.insert(0, '/home/ubuntu/sales-forecasting-mlops')

from src.config import get_settings
from src.features import FeatureEngineer
from src.data import DataProcessor
from api.services.model_service import ModelService

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Service for making predictions.
    
    Features:
    - Input preprocessing
    - Feature engineering
    - Model inference
    - Output postprocessing
    - Prediction logging
    """
    
    def __init__(self):
        """Initialize prediction service."""
        self.settings = get_settings()
        self.model_service = ModelService()
        self.feature_engineer = FeatureEngineer()
        self.data_processor = DataProcessor()
        
        logger.info("PredictionService initialized")
    
    def preprocess_input(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            input_data: Raw input DataFrame
            
        Returns:
            Preprocessed DataFrame with features
            
        Steps:
        1. Validate input
        2. Create features
        3. Handle missing values
        4. Scale features
        """
        try:
            logger.info(f"Preprocessing {len(input_data)} samples")
            
            # Create features
            df_features = self.feature_engineer.create_all_features(
                input_data, 
                target_col=None
            )
            
            # Get feature columns (exclude metadata)
            feature_cols = [col for col in df_features.columns 
                          if col not in ['date', 'sales', 'region', 'product']]
            
            X = df_features[feature_cols]
            
            # Handle missing values
            X = X.fillna(0)
            
            # Load and apply scaler
            scaler = self.data_processor.load_scaler()
            if scaler is not None:
                X_scaled = self.data_processor.scale_features(X, fit=False)
                logger.info("Features scaled")
            else:
                logger.warning("No scaler found, using unscaled features")
                X_scaled = X
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
    
    def make_prediction(
        self,
        features: pd.DataFrame,
        model_name: str,
        stage: str = "Production"
    ) -> np.ndarray:
        """
        Make prediction using specified model.
        
        Args:
            features: Preprocessed features
            model_name: Name of model to use
            stage: Model stage
            
        Returns:
            Array of predictions
        """
        try:
            logger.info(f"Making prediction with {model_name} ({stage})")
            
            # Load model
            model = self.model_service.load_model_from_registry(model_name, stage)
            
            # Make prediction
            predictions = model.predict(features)
            
            logger.info(f"Prediction complete: {len(predictions)} samples")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def postprocess_output(
        self,
        predictions: np.ndarray,
        input_data: pd.DataFrame,
        model_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Postprocess predictions into output format.
        
        Args:
            predictions: Raw predictions array
            input_data: Original input data
            model_info: Model metadata
            
        Returns:
            List of prediction dictionaries
        """
        try:
            outputs = []
            
            for i, pred in enumerate(predictions):
                output = {
                    'predicted_sales': float(pred),
                    'confidence': self._calculate_confidence(pred),
                    'model_name': model_info.get('name', 'unknown'),
                    'model_version': str(model_info.get('version', '1')),
                    'timestamp': datetime.now().isoformat(),
                }
                
                # Add input metadata if available
                if i < len(input_data):
                    output['input'] = {
                        'date': str(input_data.iloc[i]['date']),
                        'region': input_data.iloc[i].get('region', ''),
                        'product': input_data.iloc[i].get('product', ''),
                    }
                
                outputs.append(output)
            
            return outputs
            
        except Exception as e:
            logger.error(f"Postprocessing failed: {e}")
            raise
    
    def _calculate_confidence(self, prediction: float) -> float:
        """
        Calculate prediction confidence score.
        
        In production, this would use:
        - Model uncertainty estimates
        - Historical prediction accuracy
        - Input data quality metrics
        
        Args:
            prediction: Prediction value
            
        Returns:
            Confidence score (0-1)
        """
        # Placeholder: In production, implement proper confidence calculation
        # Could use prediction intervals, model uncertainty, etc.
        return 0.85
    
    def predict_single(
        self,
        input_dict: Dict[str, Any],
        model_name: str,
        stage: str = "Production"
    ) -> Dict[str, Any]:
        """
        Make single prediction.
        
        Args:
            input_dict: Input dictionary with prediction features
            model_name: Model to use
            stage: Model stage
            
        Returns:
            Prediction dictionary
            
        Example:
            >>> service = PredictionService()
            >>> result = service.predict_single({
            ...     'date': '2024-12-01',
            ...     'region': 'North',
            ...     'product': 'Electronics',
            ...     'price': 299.99,
            ...     'quantity': 50
            ... }, 'sales_forecasting_production')
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame([input_dict])
            
            # Preprocess
            features = self.preprocess_input(df)
            
            # Get model info
            model_info = self.model_service.get_model_info(model_name, stage=stage)
            
            # Predict
            predictions = self.make_prediction(features, model_name, stage)
            
            # Postprocess
            outputs = self.postprocess_output(predictions, df, model_info)
            
            return outputs[0]
            
        except Exception as e:
            logger.error(f"Single prediction failed: {e}")
            raise
    
    def predict_batch(
        self,
        input_list: List[Dict[str, Any]],
        model_name: str,
        stage: str = "Production",
        batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Make batch predictions.
        
        Args:
            input_list: List of input dictionaries
            model_name: Model to use
            stage: Model stage
            batch_size: Process in batches of this size
            
        Returns:
            List of prediction dictionaries
        """
        try:
            logger.info(f"Batch prediction: {len(input_list)} samples")
            
            # Convert to DataFrame
            df = pd.DataFrame(input_list)
            
            # Preprocess
            features = self.preprocess_input(df)
            
            # Get model info
            model_info = self.model_service.get_model_info(model_name, stage=stage)
            
            # Predict in batches
            all_predictions = []
            for i in range(0, len(features), batch_size):
                batch_features = features.iloc[i:i+batch_size]
                batch_predictions = self.make_prediction(batch_features, model_name, stage)
                all_predictions.extend(batch_predictions)
            
            # Postprocess
            outputs = self.postprocess_output(
                np.array(all_predictions), 
                df, 
                model_info
            )
            
            return outputs
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def log_prediction(
        self,
        prediction: Dict[str, Any],
        store_in_db: bool = True
    ) -> bool:
        """
        Log prediction to database.
        
        Args:
            prediction: Prediction dictionary
            store_in_db: Whether to store in database
            
        Returns:
            Success status
        """
        try:
            if not store_in_db:
                return True
            
            from sqlalchemy import create_engine
            
            engine = create_engine(self.settings.get_database_url())
            
            # Prepare data for database
            db_record = {
                'timestamp': datetime.now(),
                'predicted_sales': prediction['predicted_sales'],
                'model_name': prediction['model_name'],
                'model_version': prediction['model_version'],
                'confidence': prediction.get('confidence'),
            }
            
            # Add input data if available
            if 'input' in prediction:
                db_record.update(prediction['input'])
            
            # Insert into database
            df = pd.DataFrame([db_record])
            df.to_sql('predictions', engine, if_exists='append', index=False)
            
            engine.dispose()
            
            logger.info("Prediction logged to database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
            return False
    
    def get_prediction_stats(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get prediction statistics for last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Statistics dictionary
        """
        try:
            from sqlalchemy import create_engine
            
            engine = create_engine(self.settings.get_database_url())
            
            query = f"""
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(predicted_sales) as avg_prediction,
                    MIN(predicted_sales) as min_prediction,
                    MAX(predicted_sales) as max_prediction,
                    AVG(confidence) as avg_confidence
                FROM predictions
                WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
            """
            
            df = pd.read_sql(query, engine)
            engine.dispose()
            
            if df.empty:
                return {
                    'total_predictions': 0,
                    'avg_prediction': 0,
                    'period_hours': hours
                }
            
            stats = df.iloc[0].to_dict()
            stats['period_hours'] = hours
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}
