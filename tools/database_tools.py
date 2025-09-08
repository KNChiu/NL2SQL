import sqlite3
from typing import List, Dict, Any, Optional
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from contextlib import contextmanager
import logging

from config.settings import settings


class DatabaseManager:
    """Enhanced database manager with safety controls"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or settings.database.database_url
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Simplify SQLite path handling
        if self.database_url.startswith('sqlite:///'):
            self._resolve_sqlite_path()
            
        self.logger.debug(f"Initializing database with URL: {self.database_url}")
        
        # Create test database if it doesn't exist and it's SQLite
        if self.database_url.startswith('sqlite:///'):
            self._ensure_test_database_exists()
        
        try:
            self.db = SQLDatabase.from_uri(self.database_url)
            self.logger.debug("SQLDatabase initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLDatabase: {e}")
            raise
        
    def get_database(self) -> SQLDatabase:
        """Get the SQLDatabase instance"""
        return self.db
    
    def list_tables(self) -> List[str]:
        """Get list of all tables in the database"""
        try:
            self.logger.debug("Attempting to get usable table names")
            result = self.db.get_usable_table_names()
            self.logger.debug(f"Successfully retrieved {len(result) if result else 0} tables: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error listing tables: {e}", exc_info=True)
            return []
    
    def get_table_schema(self, table_names: List[str]) -> str:
        """Get schema for specified tables"""
        try:
            self.logger.debug(f"Getting schema for tables: {table_names}")
            result = self.db.get_table_info(table_names=table_names)
            self.logger.debug(f"Successfully retrieved schema (length: {len(result) if result else 0})")
            self.logger.debug(f"Schema content: {result[:500]}..." if result and len(result) > 500 else f"Schema content: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error getting table schema for {table_names}: {e}", exc_info=True)
            return ""
    
    def execute_safe_query(self, query: str) -> Optional[str]:
        """Execute query with safety checks"""
        try:
            # Additional safety check
            if not query.strip().upper().startswith('SELECT'):
                raise ValueError("Only SELECT queries are allowed")
            
            result = self.db.run_no_throw(query)
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return None
    
    def get_sample_data(self, table_name: str, limit: int = 3) -> Optional[str]:
        """Get sample data from a table"""
        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            return self.execute_safe_query(query)
        except Exception as e:
            self.logger.error(f"Error getting sample data: {e}")
            return None
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = None
        try:
            if self.database_url.startswith('sqlite:'):
                db_path = self.database_url.replace('sqlite:///', '')
                conn = sqlite3.connect(db_path, timeout=settings.database.connection_timeout)
                yield conn
            else:
                # For other databases, use SQLDatabase connection
                yield self.db._engine.connect()
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            self.logger.debug("Testing database connection...")
            self.logger.debug(f"Database URL: {self.database_url}")
            
            # Test basic connection
            try:
                engine = self.db._engine
                with engine.connect() as conn:
                    self.logger.debug("Database engine connection successful")
            except Exception as conn_e:
                self.logger.error(f"Database engine connection failed: {conn_e}")
                return False
            
            # Test table listing
            tables = self.list_tables()
            table_count = len(tables) if tables else 0
            self.logger.debug(f"Connection test result: {table_count} tables found")
            
            return table_count > 0
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}", exc_info=True)
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive database information"""
        info = {
            "database_url": self.database_url,
            "connection_status": "disconnected",
            "tables": [],
            "total_tables": 0
        }
        
        try:
            tables = self.list_tables()
            info.update({
                "connection_status": "connected",
                "tables": tables,
                "total_tables": len(tables)
            })
            
            # Get table details
            table_details = {}
            for table in tables[:5]:  # Limit to first 5 tables for performance
                try:
                    schema = self.get_table_schema([table])
                    sample = self.get_sample_data(table, limit=2)
                    table_details[table] = {
                        "schema": schema,
                        "sample_data": sample
                    }
                except Exception as e:
                    self.logger.warning(f"Could not get details for table {table}: {e}")
            
            info["table_details"] = table_details
            
        except Exception as e:
            self.logger.error(f"Error getting database info: {e}")
            info["error"] = str(e)
        
        return info
    
    def _resolve_sqlite_path(self):
        """Resolve SQLite database path to absolute path"""
        try:
            # Extract database file path from URL
            db_path = self.database_url.replace('sqlite:///', '')
            
            # Convert to Path object for easier handling
            from pathlib import Path
            path_obj = Path(db_path)
            
            # If path is not absolute, make it relative to project root
            if not path_obj.is_absolute():
                project_root = Path(__file__).parent.parent
                absolute_path = project_root / db_path
                self.database_url = f'sqlite:///{absolute_path}'
                self.logger.debug(f"Resolved relative path to: {absolute_path}")
            else:
                self.logger.debug(f"Using absolute path: {path_obj}")
                
        except Exception as e:
            self.logger.error(f"Error resolving SQLite path: {e}")
            # Fall back to original URL
            pass
    
    def _ensure_test_database_exists(self):
        """Create a test database with sample data if it doesn't exist"""
        
        from pathlib import Path
        import sqlite3
        
        try:
            # Extract path from URL
            db_path = self.database_url.replace('sqlite:///', '')
            db_file = Path(db_path)
            
            if not db_file.exists():
                self.logger.info(f"Creating test database at: {db_path}")
                
                # Create directory if needed
                db_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Create test database with sample data
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                try:
                    # Create sample tables
                    cursor.execute('''
                        CREATE TABLE users (
                            id INTEGER PRIMARY KEY,
                            name TEXT NOT NULL,
                            email TEXT UNIQUE,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    cursor.execute('''
                        CREATE TABLE orders (
                            id INTEGER PRIMARY KEY,
                            user_id INTEGER,
                            product_name TEXT NOT NULL,
                            price DECIMAL(10,2),
                            order_date DATE DEFAULT CURRENT_DATE,
                            FOREIGN KEY (user_id) REFERENCES users (id)
                        )
                    ''')
                    
                    # Insert sample data
                    cursor.execute('''
                        INSERT INTO users (name, email) VALUES 
                        ('Alice Johnson', 'alice@example.com'),
                        ('Bob Smith', 'bob@example.com'),
                        ('Carol Brown', 'carol@example.com')
                    ''')
                    
                    cursor.execute('''
                        INSERT INTO orders (user_id, product_name, price) VALUES 
                        (1, 'Laptop', 999.99),
                        (1, 'Mouse', 29.99),
                        (2, 'Keyboard', 79.99),
                        (3, 'Monitor', 299.99),
                        (2, 'Webcam', 89.99)
                    ''')
                    
                    conn.commit()
                    self.logger.info("Test database created successfully with sample data")
                    
                except Exception as db_error:
                    self.logger.error(f"Error creating test database: {db_error}")
                    conn.rollback()
                    
                finally:
                    conn.close()
            
        except Exception as e:
            self.logger.error(f"Error in _ensure_test_database_exists: {e}")


class QueryOptimizer:
    """Query optimization utilities"""
    
    @staticmethod
    def add_limit_if_missing(query: str, default_limit: int = 100) -> str:
        """Add LIMIT clause if missing"""
        query = query.strip()
        
        # Check if LIMIT already exists
        if 'LIMIT' in query.upper():
            return query
            
        # Add LIMIT before ORDER BY if it exists
        if 'ORDER BY' in query.upper():
            order_index = query.upper().rfind('ORDER BY')
            return f"{query[:order_index]}LIMIT {default_limit} {query[order_index:]}"
        
        # Add LIMIT at the end
        return f"{query} LIMIT {default_limit}"
    
    @staticmethod
    def optimize_for_preview(query: str) -> str:
        """Optimize query for preview/testing"""
        
        # Add limit for safety
        query = QueryOptimizer.add_limit_if_missing(query, 10)
        
        # Remove potentially expensive operations for preview
        # This is a simplified version - more sophisticated optimization can be added
        
        return query
    
    @staticmethod
    def estimate_query_cost(query: str) -> str:
        """Estimate query complexity (simplified)"""
        
        query_upper = query.upper()
        cost_factors = []
        
        if 'JOIN' in query_upper:
            join_count = query_upper.count('JOIN')
            cost_factors.append(f"{join_count} joins")
        
        if 'GROUP BY' in query_upper:
            cost_factors.append("grouping")
        
        if 'ORDER BY' in query_upper:
            cost_factors.append("sorting")
        
        if 'HAVING' in query_upper:
            cost_factors.append("filtering after grouping")
        
        if not cost_factors:
            return "low"
        elif len(cost_factors) == 1:
            return "medium"
        else:
            return "high"


# Global database manager instance
db_manager = DatabaseManager()