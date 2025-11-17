"""Utility functions for the stock analysis system"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import config


def setup_logging():
    """Setup logging configuration"""
    log_file = config.LOGS_DIR / f"nepse_analysis_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def save_json(data: dict, filename: str, directory: str = None):
    """Save data to JSON file"""
    if directory is None:
        directory = config.REPORTS_DIR
    
    filepath = Path(directory) / filename
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return filepath


def load_json(filepath: str):
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_currency(amount: float, currency: str = 'NPR'):
    """Format currency value"""
    return f"{currency} {amount:,.2f}"


def format_percentage(value: float):
    """Format percentage value"""
    return f"{value:.2f}%"


def calculate_percentage_change(old_value: float, new_value: float):
    """Calculate percentage change"""
    if old_value == 0:
        return 0
    
    return ((new_value - old_value) / abs(old_value)) * 100


logger = setup_logging()
