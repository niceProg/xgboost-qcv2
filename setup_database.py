#!/usr/bin/env python3
"""
Setup script for XGBoost training database.
Creates the database and tables needed for storing training artifacts.
"""

import os
import sys
from pathlib import Path
import logging
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from database_storage import Base, DatabaseStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_database():
    """Create the database if it doesn't exist."""
    load_dotenv()

    # Database configuration (connect without specifying database first)
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 3306)),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', '')
    }

    db_name = os.getenv('DB_NAME', 'xgboost_training')

    try:
        # Connect to MySQL server (without database)
        connection_string = (
            f"mysql+pymysql://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}"
        )

        engine = create_engine(connection_string)

        with engine.connect() as conn:
            # Create database if not exists
            conn.execute(f"CREATE DATABASE IF NOT EXISTS {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            logger.info(f"Database '{db_name}' created or already exists")

    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        raise

def create_tables():
    """Create all tables in the database."""
    # Initialize database storage to create tables
    db_storage = DatabaseStorage()

    # Tables are created automatically during initialization
    logger.info("All tables created successfully")

def verify_setup():
    """Verify the database setup by checking tables exist."""
    try:
        db_storage = DatabaseStorage()

        # Test connection and get table info
        with db_storage.get_session() as db:
            # Get all table names
            result = db.execute("SHOW TABLES")
            tables = [row[0] for row in result.fetchall()]

            logger.info("Tables created:")
            for table in sorted(tables):
                logger.info(f"  - {table}")

            # Test session creation
            session_id = db_storage.create_training_session(
                notes="Database setup verification"
            )
            logger.info(f"Test session created: {session_id}")

            return True

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False

def main():
    """Main setup function."""
    logger.info("Starting database setup...")

    try:
        # Step 1: Create database
        logger.info("Step 1: Creating database...")
        create_database()

        # Step 2: Create tables
        logger.info("Step 2: Creating tables...")
        create_tables()

        # Step 3: Verify setup
        logger.info("Step 3: Verifying setup...")
        if verify_setup():
            logger.info("\n=== Database Setup Complete ===")
            logger.info("Database is ready for use!")
            logger.info("\nNext steps:")
            logger.info("1. Run load_database.py to load trading data")
            logger.info("2. Run merge_7_tables.py to merge the tables")
            logger.info("3. Run feature_engineering.py to create features")
            logger.info("4. Run label_builder.py to create labels")
            logger.info("5. Run xgboost_trainer.py to train model")
            logger.info("6. Run model_evaluation_with_leverage.py to evaluate")
        else:
            logger.error("Setup verification failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()