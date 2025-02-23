import logging
from api import app
import sys
import os

import rag.db

# Configure logging to output to both console and file
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

def run_app():
    """
    Function to run the Flask application with custom settings and error handling.
    """
    try:
        # Default configuration for host and port
        host = os.getenv("FLASK_HOST", "0.0.0.0")
        port = int(os.getenv("FLASK_PORT", 4399))

        logger.info(f"Starting Flask app on {host}:{port}...")
        app.run(debug=True, host=host, port=port)

    except Exception as e:
        logger.error(f"An error occurred while starting the application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # import rag
    # rag.db.test()
    run_app()
