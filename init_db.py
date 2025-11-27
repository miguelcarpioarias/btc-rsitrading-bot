#!/usr/bin/env python
"""
Database initialization script for Render deployment.
Run this before starting the web server to ensure tables are created.
"""
import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def init_database():
    """Initialize the database"""
    try:
        from database import init_db
        
        database_url = os.getenv('DATABASE_URL')
        
        if not database_url:
            logger.error("DATABASE_URL environment variable not set")
            sys.exit(1)
        
        logger.info(f"Initializing database at: {database_url[:50]}...")
        engine = init_db()
        logger.info("✅ Database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    logger.info("Starting database initialization...")
    init_database()
    logger.info("Database initialization complete!")
