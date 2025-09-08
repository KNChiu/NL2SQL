# NL2SQL LangGraph Workflow

A modular, secure, and extensible Natural Language to SQL system built with LangGraph for workflow orchestration.

## 🎯 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Security First**: Built-in SQL injection prevention and query validation
- **LangGraph Orchestration**: Reliable workflow management with retries and error handling
- **Structured JSON Outputs**: Comprehensive state tracking and result formatting
- **Multi-turn Interaction**: Support for iterative query refinement
- **Extensible Design**: Easy to add new validation rules and database connectors

## 📁 Project Structure

```
NL2SQL/
├── main.py                     # Main execution entry point
├── config/
│   ├── __init__.py
│   └── settings.py             # Configuration management
├── core/
│   ├── __init__.py
│   ├── state.py                # State management and JSON outputs
│   └── workflow.py             # LangGraph workflow definition
├── security/
│   ├── __init__.py
│   └── validator.py            # SQL security validation
├── agents/                     # Future: Specialized agent nodes
├── tools/
│   ├── __init__.py
│   └── database_tools.py       # Database utilities and tools
├── utils/
│   ├── __init__.py
│   └── helpers.py              # Utility functions
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## 🚀 Quick Start

### Option 1: Core Installation (Recommended)
```bash
# Install minimal required dependencies
pip install -r requirements-core.txt
# or using uv
uv pip install -r requirements-core.txt
```

### Option 2: Full Installation
```bash
# Install all dependencies (including optional ones)
pip install -r requirements.txt
# or using uv  
uv pip install -r requirements.txt
```

### Setup and Run

1. **Set Up Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key and database settings
   ```

2. **Run Interactive Mode**
   ```bash
   python main.py --interactive
   ```

3. **Single Query**
   ```bash
   python main.py --query "Show me the top 5 customers by total sales"
   ```

### Installation Troubleshooting

**If you encounter package installation issues:**

1. **Use requirements-core.txt** for minimal installation
2. **Check Python version**: Requires Python 3.8+
3. **SQLite is built-in**: No need to install sqlite3 separately
4. **For specific databases**: Install drivers separately:
   ```bash
   pip install psycopg2-binary  # PostgreSQL
   pip install pymysql          # MySQL
   ```

## 🔧 Configuration

Key environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: Model to use (default: gpt-3.5-turbo)
- `DATABASE_URL`: Database connection string (default: sqlite:///Chinook.db)
- `DEBUG`: Enable debug logging (default: False)

## 🛡️ Security Features

- **Query Type Restriction**: Only SELECT queries allowed
- **Pattern Validation**: Blocks dangerous SQL operations (INSERT, UPDATE, DELETE, DROP, etc.)
- **Syntax Checking**: Basic SQL syntax validation
- **Result Limiting**: Automatic LIMIT clauses for safety
- **Error Handling**: Graceful failure with informative messages

## 🔄 Workflow Stages

The system follows a clear workflow with structured state management:

1. **INITIALIZED**: Starting point with user query
2. **SCHEMA_RETRIEVED**: Database schema information gathered
3. **QUERY_GENERATED**: SQL query generated from natural language
4. **QUERY_VALIDATED**: Security and correctness validation completed
5. **QUERY_EXECUTED**: SQL execution with results
6. **RESPONSE_GENERATED**: Natural language response created

## 📊 JSON State Output

Each workflow execution produces structured JSON output:

```json
{
  "workflow_id": "nl2sql_20241203_143022",
  "user_query": "Show me top customers",
  "current_stage": "RESPONSE_GENERATED", 
  "status": "success",
  "query_info": {
    "sql_query": "SELECT * FROM customers ORDER BY total_sales DESC LIMIT 5",
    "validation_status": "QUERY_IS_CORRECT",
    "retry_count": 0
  },
  "response": {
    "natural_language": "Here are the top 5 customers by sales...",
    "structured_data": {...},
    "confidence": 0.9
  },
  "errors": [],
  "execution_time": 2.34
}
```

## 🔄 Error Handling & Retries

- Automatic retry for failed operations (max 3 attempts)
- Detailed error categorization and logging
- Graceful degradation with informative error messages
- State preservation across retry attempts

## 🧪 Usage Examples

**Basic Query:**
```bash
python main.py --query "What are the most popular music genres?"
```

**SQL Validation:**
```bash
python main.py --validate-sql "SELECT * FROM users"
```

**Database Info:**
```bash
python main.py --db-info
```

**Interactive Mode:**
```bash
python main.py --interactive
```

## 🛠️ Development

### Adding New Validation Rules

Edit `security/validator.py` and add patterns to `DANGEROUS_PATTERNS` or `COMMON_ERRORS`.

### Extending Workflow Nodes

Add new nodes to `core/workflow.py` and update the workflow graph structure.

### Custom Database Connectors

Extend `tools/database_tools.py` with new database type support.

## 📋 Requirements

- Python 3.8+
- OpenAI API key
- SQLite database (Chinook.db included as example)
- See `requirements.txt` for full dependency list

## 🤝 Contributing

1. Follow the existing modular architecture
2. Add appropriate error handling and logging
3. Include security validation for new features
4. Update documentation and examples

## 📝 License

This project is built for educational and demonstration purposes.