from sqlalchemy import create_engine
from urllib.parse import quote_plus

DB_USER = "postgres"
DB_PASSWORD = "Postgres@123"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "diabetes_readmission"

def get_engine():
    """
    Returns a SQLAlchemy engine — the object every other script uses
    to read from or write to Postgres. Created once per script run.
    """
    encoded_password = quote_plus(DB_PASSWORD)  # safely escapes special characters like '@'
    connection_string = (
        f"postgresql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return create_engine(connection_string)