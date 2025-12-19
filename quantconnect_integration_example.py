#!/usr/bin/env python3
"""
QuantConnect Integration Example for XGBoost Model API
This example shows how to use the API to fetch latest models and dataset summaries
instead of manual ObjectStore uploads
"""

import base64
import json
import requests
import pickle
import os
from datetime import datetime

# API Configuration
API_BASE_URL = "http://your-api-server:5000"  # Replace with your actual API server URL
API_TIMEOUT = 30  # seconds

class XGBoostQuantConnectAPI:
    """API client for XGBoost model integration with QuantConnect."""

    def __init__(self, api_base_url: str = API_BASE_URL, timeout: int = API_TIMEOUT):
        self.api_base_url = api_base_url.rstrip('/')
        self.timeout = timeout

    def _make_request(self, endpoint: str) -> dict:
        """Make GET request to API endpoint."""
        url = f"{self.api_base_url}{endpoint}"

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")

    def get_latest_model(self) -> dict:
        """Get latest trained model."""
        return self._make_request('/api/v1/latest/model')

    def get_latest_dataset_summary(self) -> dict:
        """Get latest dataset summary."""
        return self._make_request('/api/v1/latest/dataset-summary')

    def get_model_by_session(self, session_id: str) -> dict:
        """Get model by specific session ID."""
        return self._make_request(f'/api/v1/model/{session_id}')

    def get_dataset_summary_by_session(self, session_id: str) -> dict:
        """Get dataset summary by specific session ID."""
        return self._make_request(f'/api/v1/dataset-summary/{session_id}')

    def list_sessions(self) -> dict:
        """List all available training sessions."""
        return self._make_request('/api/v1/sessions')

    def decode_base64_model(self, model_base64: str) -> object:
        """Decode base64 model string back to Python object."""
        try:
            model_bytes = base64.b64decode(model_base64)
            model = pickle.loads(model_bytes)
            return model
        except Exception as e:
            raise Exception(f"Failed to decode model: {e}")

    def decode_base64_text(self, text_base64: str) -> str:
        """Decode base64 text string."""
        try:
            text_bytes = base64.b64decode(text_base64)
            return text_bytes.decode('utf-8')
        except Exception as e:
            raise Exception(f"Failed to decode text: {e}")

# QuantConnect Algorithm Example
class QuantConnectAlgorithm:
    """
    Example of how to integrate XGBoost API into QuantConnect algorithm
    """

    def initialize(self):
        """Initialize algorithm with XGBoost model from API."""

        # Initialize API client
        self.api_client = XGBoostQuantConnectAPI(
            api_base_url="http://your-xgboost-api-server:5000"
        )

        # Fetch latest model and dataset summary
        try:
            # Get latest model
            model_response = self.api_client.get_latest_model()

            if model_response['success']:
                # Decode model from base64
                self.xgb_model = self.api_client.decode_base64_model(
                    model_response['model_data_base64']
                )

                self.model_session_id = model_response['session_id']
                self.model_version = model_response['model_version']

                self.Debug(f"✅ Loaded XGBoost model from session: {self.model_session_id}")
                self.Debug(f"📅 Model created at: {model_response['created_at']}")
                self.Debug(f"🔢 Feature names: {model_response['feature_names']}")

            else:
                self.Error(f"❌ Failed to load model: {model_response.get('error')}")
                return

            # Get latest dataset summary
            summary_response = self.api_client.get_latest_dataset_summary()

            if summary_response['success']:
                # Decode dataset summary from base64
                summary_text = self.api_client.decode_base64_text(
                    summary_response['summary_data_base64']
                )

                self.Debug(f"📊 Dataset summary loaded for session: {summary_response['session_id']}")
                self.Debug(f"📄 Summary file: {summary_response['summary_file']}")

                # Parse dataset summary (example)
                self.parse_dataset_summary(summary_text)

            else:
                self.Error(f"❌ Failed to load dataset summary: {summary_response.get('error')}")

        except Exception as e:
            self.Error(f"❌ Failed to initialize XGBoost model: {e}")

    def parse_dataset_summary(self, summary_text: str):
        """Parse dataset summary text and extract useful information."""
        lines = summary_text.split('\n')

        for line in lines:
            if 'Total samples:' in line:
                total_samples = line.split(':')[1].strip().replace(',', '')
                self.Debug(f"📈 Total training samples: {total_samples}")

            elif 'Bullish (1):' in line:
                bullish_count = line.split(':')[1].strip().split(' ')[0].replace(',', '')
                self.Debug(f"📈 Bullish samples: {bullish_count}")

            elif 'Bearish (0):' in line:
                bearish_count = line.split(':')[1].strip().split(' ')[0].replace(',', '')
                self.Debug(f"📉 Bearish samples: {bearish_count}")

            elif 'Exchange:' in line:
                exchanges = line.split(':')[1].strip()
                self.Debug(f"🏢 Exchanges: {exchanges}")

            elif 'Symbols:' in line:
                symbols = line.split(':')[1].strip()
                self.Debug(f"💱 Symbols: {symbols}")

    def predict_with_model(self, features: dict) -> float:
        """Make prediction using loaded XGBoost model."""
        if not hasattr(self, 'xgb_model'):
            self.Error("❌ XGBoost model not loaded")
            return 0.0

        try:
            # Convert features to the format expected by the model
            # This depends on how your features are structured
            import pandas as pd

            # Example: convert dict to DataFrame with correct column order
            feature_df = pd.DataFrame([features])

            # Make prediction
            prediction = self.xgb_model.predict_proba(feature_df)[:, 1][0]

            return float(prediction)

        except Exception as e:
            self.Error(f"❌ Prediction failed: {e}")
            return 0.0

# Example usage outside QuantConnect
def example_usage():
    """Example of how to use the API client."""

    # Initialize API client
    api_client = XGBoostQuantConnectAPI()

    try:
        # Check health
        health = api_client._make_request('/health')
        print(f"🏥 API Health: {health}")

        # List all sessions
        sessions = api_client.list_sessions()
        print(f"📋 Available sessions: {sessions['total_sessions']}")

        # Get latest model
        model = api_client.get_latest_model()
        print(f"🤖 Latest model: {model['model_name']} (Session: {model['session_id']})")

        # Decode and inspect model
        xgb_model = api_client.decode_base64_model(model['model_data_base64'])
        print(f"🔧 Model type: {type(xgb_model)}")

        # Get dataset summary
        summary = api_client.get_latest_dataset_summary()
        summary_text = api_client.decode_base64_text(summary['summary_data_base64'])
        print(f"📊 Dataset summary preview:\n{summary_text[:500]}...")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    example_usage()