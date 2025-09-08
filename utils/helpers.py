import json
import logging
from typing import Any, Dict, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path


def setup_logging(debug: bool = False) -> logging.Logger:
    """Set up enhanced logging configuration"""
    
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create formatter with more detailed information
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)-20s | %(levelname)-8s | %(funcName)-15s:%(lineno)-3d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with color support
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    
    # File handler for persistent logging
    file_handler = logging.FileHandler('nl2sql.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # Always log debug to file
    
    # Configure root logger
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Suppress verbose third-party loggers
    third_party_loggers = [
        'httpx',
        'httpcore', 
        'urllib3',
        'requests',
        'openai',
        'langchain',
        'langsmith'
    ]
    
    for logger_name in third_party_loggers:
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.setLevel(logging.WARNING)  # Only show warnings and errors
    
    # Create and return application logger
    app_logger = logging.getLogger('nl2sql')
    app_logger.info(f"Logging initialized - Debug: {debug}, Level: {logging.getLevelName(log_level)}")
    
    return app_logger


def format_sql_result(result: str, max_length: int = 1000) -> str:
    """Format SQL result for display"""
    
    if not result:
        return "No results found."
    
    # Truncate if too long
    if len(result) > max_length:
        return result[:max_length] + "... (truncated)"
    
    return result


def parse_json_safely(json_str: str) -> Dict[str, Any]:
    """Safely parse JSON string"""
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format", "raw_content": json_str}


def save_workflow_result(workflow_id: str, result: Dict[str, Any]) -> bool:
    """Save workflow result to file"""
    
    try:
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Save result with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"workflow_{workflow_id}_{timestamp}.json"
        filepath = results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        
        return True
        
    except Exception as e:
        logging.error(f"Error saving workflow result: {e}")
        return False


def load_workflow_result(workflow_id: str) -> Dict[str, Any]:
    """Load the most recent workflow result for given ID"""
    
    try:
        results_dir = Path("results")
        if not results_dir.exists():
            return {}
        
        # Find the most recent file for this workflow_id
        pattern = f"workflow_{workflow_id}_*.json"
        files = list(results_dir.glob(pattern))
        
        if not files:
            return {}
        
        # Get the most recent file
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    except Exception as e:
        logging.error(f"Error loading workflow result: {e}")
        return {}


def clean_old_results(days_old: int = 7) -> int:
    """Clean up old result files"""
    
    try:
        results_dir = Path("results")
        if not results_dir.exists():
            return 0
        
        cutoff_time = datetime.now() - timedelta(days=days_old)
        deleted_count = 0
        
        for file in results_dir.glob("workflow_*.json"):
            if datetime.fromtimestamp(file.stat().st_mtime) < cutoff_time:
                file.unlink()
                deleted_count += 1
        
        return deleted_count
        
    except Exception as e:
        logging.error(f"Error cleaning old results: {e}")
        return 0


def validate_user_input(user_query: str) -> Tuple[bool, str]:
    """Validate user input"""
    
    if not user_query or not user_query.strip():
        return False, "Query cannot be empty"
    
    # Check for reasonable length
    if len(user_query.strip()) < 3:
        return False, "Query is too short"
    
    if len(user_query) > 1000:
        return False, "Query is too long (max 1000 characters)"
    
    # Check for obvious malicious patterns
    dangerous_patterns = [
        'javascript:', 'data:', '<script', 'eval(', 'exec('
    ]
    
    query_lower = user_query.lower()
    for pattern in dangerous_patterns:
        if pattern in query_lower:
            return False, f"Query contains potentially dangerous content: {pattern}"
    
    return True, "Valid input"


def extract_table_names_from_query(query: str) -> List[str]:
    """Extract table names from SQL query (simple regex-based approach)"""
    
    import re
    
    # Simple pattern to find table names after FROM and JOIN
    patterns = [
        r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'\bINTO\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        r'\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    ]
    
    tables = set()
    
    for pattern in patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        tables.update(matches)
    
    return list(tables)


def format_execution_time(seconds: float) -> str:
    """Format execution time in human readable format"""
    
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def create_error_response(
    error_message: str, 
    error_type: str = "GeneralError",
    details: Dict[str, Any] = None,
    suggestions: List[str] = None
) -> Dict[str, Any]:
    """Create enhanced standardized error response"""
    
    error_response = {
        "success": False,
        "error": {
            "type": error_type,
            "message": format_error_message(error_message, error_type),
            "timestamp": datetime.now().isoformat(),
            "error_id": f"{error_type}_{int(datetime.now().timestamp() * 1000)}"
        },
        "data": None
    }
    
    # Add optional details
    if details:
        error_response["error"]["details"] = details
    
    # Add suggestions for resolution
    if suggestions:
        error_response["error"]["suggestions"] = suggestions
    elif error_type in ERROR_TYPE_SUGGESTIONS:
        error_response["error"]["suggestions"] = ERROR_TYPE_SUGGESTIONS[error_type]
    
    return error_response

def format_error_message(message: str, error_type: str) -> str:
    """Format error message with context"""
    
    # Add context based on error type
    prefixes = {
        "InputValidationError": "⚠️ Input validation failed",
        "WorkflowExecutionError": "🔄 Workflow execution error",
        "QueryGenerationError": "🛠️ SQL generation failed", 
        "QueryValidationError": "🔍 SQL validation failed",
        "QueryExecutionError": "⚡ Query execution failed",
        "DatabaseConnectionError": "🗄️ Database connection error",
        "SchemaRetrievalError": "📋 Schema retrieval failed",
        "SecurityError": "🔒 Security validation failed"
    }
    
    prefix = prefixes.get(error_type, "❌ Error")
    return f"{prefix}: {message}"

# Predefined suggestions for common error types
ERROR_TYPE_SUGGESTIONS = {
    "InputValidationError": [
        "Ensure your query is not empty and within length limits",
        "Remove any special characters or comments",
        "Try rephrasing your natural language query"
    ],
    "QueryGenerationError": [
        "Try rephrasing your question in simpler terms",
        "Be more specific about what data you want to see",
        "Check if the table names exist in the database"
    ],
    "QueryValidationError": [
        "Ensure your query only uses SELECT statements",
        "Remove any INSERT, UPDATE, DELETE operations",
        "Check for proper SQL syntax"
    ],
    "QueryExecutionError": [
        "Check if the table and column names are correct",
        "Verify your WHERE clause conditions",
        "Make sure you have proper permissions"
    ],
    "DatabaseConnectionError": [
        "Check if the database file exists",
        "Verify the database URL in configuration",
        "Ensure proper database permissions"
    ],
    "SecurityError": [
        "Remove any suspicious patterns from your query",
        "Avoid using comments or special characters", 
        "Stick to simple SELECT statements with WHERE clauses"
    ]
}


def create_success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
    """Create enhanced standardized success response"""
    
    response = {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add execution metadata if available
    if isinstance(data, dict) and "execution_time" in data:
        response["execution_time"] = data.get("execution_time")
        response["execution_time_formatted"] = format_execution_time(data.get("execution_time", 0))
    
    return response

def log_function_call(func_name: str, args: Dict[str, Any] = None, logger: logging.Logger = None):
    """Log function call with arguments for debugging"""
    
    if logger is None:
        logger = logging.getLogger('nl2sql')
    
    args_str = ""
    if args:
        # Sanitize sensitive information
        safe_args = {}
        for key, value in args.items():
            if key.lower() in ['password', 'token', 'key', 'secret']:
                safe_args[key] = "***HIDDEN***"
            elif isinstance(value, str) and len(value) > 100:
                safe_args[key] = value[:100] + "..."
            else:
                safe_args[key] = value
        args_str = f" with args: {safe_args}"
    
    logger.debug(f"Calling function: {func_name}{args_str}")

def log_execution_context(context: Dict[str, Any], logger: logging.Logger = None):
    """Log execution context for debugging"""
    
    if logger is None:
        logger = logging.getLogger('nl2sql')
    
    logger.debug("Execution context:")
    for key, value in context.items():
        if isinstance(value, str) and len(value) > 200:
            logger.debug(f"  {key}: {value[:200]}...")
        else:
            logger.debug(f"  {key}: {value}")

def get_user_friendly_error(error: Exception) -> str:
    """Convert technical errors to user-friendly messages"""
    
    error_type = type(error).__name__
    error_message = str(error).lower()
    
    # Database connection errors
    if "connection" in error_message or "database" in error_message:
        return "Unable to connect to the database. Please check your database configuration."
    
    # SQL syntax errors
    if "syntax" in error_message or "parse" in error_message:
        return "There was a problem with the SQL query syntax. Please try rephrasing your question."
    
    # Permission errors
    if "permission" in error_message or "access" in error_message:
        return "Access denied. You don't have permission to perform this operation."
    
    # Timeout errors
    if "timeout" in error_message:
        return "The operation timed out. Please try a simpler query or check your connection."
    
    # Network errors
    if "network" in error_message or "connection refused" in error_message:
        return "Network connection problem. Please check your internet connection."
    
    # File not found errors
    if "file not found" in error_message or "no such file" in error_message:
        return "Required file not found. Please check the file path and ensure the file exists."
    
    # Generic error with some context
    return f"An unexpected error occurred ({error_type}). Please try again or contact support if the problem persists."

def create_debug_info(workflow_state: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create debug information for troubleshooting"""
    
    debug_info = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "python_version": "",  # Would need sys.version in real implementation
            "platform": "",        # Would need platform.system() in real implementation
        }
    }
    
    if workflow_state:
        debug_info["workflow_state"] = {
            "current_stage": workflow_state.get("current_stage"),
            "errors_count": len(workflow_state.get("errors", [])),
            "retry_count": workflow_state.get("query_info", {}).get("retry_count", 0),
            "has_validation_result": workflow_state.get("query_info", {}).get("validation_result") is not None
        }
    
    return debug_info