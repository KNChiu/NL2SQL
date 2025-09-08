## 1. Introduction

Develop a system that converts natural language queries into SQL and executes them. The workflow will be orchestrated using **LangGraph**, with an emphasis on simplicity, traceability, multi-turn interaction, and security.

## 2. Objectives

* Modular and extensible design
* Reliable SQL validation and execution results
* Enforce query safety (read-only, prevent dangerous operations)

## 3. Functional Requirements

### 3.1 Natural Language to SQL

* Accept user input in natural language and call an LLM to generate SQL queries
* Use structured prompts to ensure the output is strictly SQL

### 3.2 SQL Validation (Reflection)

* Validate the correctness of the generated SQL; provide corrections if invalid or unreasonable
* Use structured JSON status outputs (e.g., `QUERY_IS_CORRECT`, `QUERY_IS_NOT_CORRECT`)

### 3.3 SQL Execution and Result Delivery

* Execute only `SELECT` queries
* Return results in a human-readable text format or as structured JSON

### 3.4 LangGraph Workflow

* Define **State** and **Node** flows (Generate → Validate → Execute → Respond)
* Allow retries in case of errors during the workflow

### 3.5 Security Controls

* Block `DML` and `DDL` operations
* Apply system prompts and validation to prevent unsafe actions

## 4. Implementation Status

✅ **Completed Features:**
- Modular architecture with dedicated security, core, tools modules
- LangGraph workflow with StateGraph implementation
- Programmatic SQL validation beyond prompt-based security
- Structured JSON state outputs throughout workflow
- Error handling with retry logic (max 3 attempts)
- Updated to use `get_usable_table_names()` API
- Interactive CLI interface with multiple operation modes
- Comprehensive configuration management with Pydantic
- Structured logging and result persistence
- **NEW**: Execution tracing and workflow visualization capabilities
- **NEW**: GraphRecursionError resolution with improved retry logic
- **NEW**: LangGraph workflow structure visualization (`--show-graph`)
- **NEW**: Real-time execution path tracking (`--trace`)
- **NEW**: Enhanced state management for trace mode compatibility
- **NEW**: Improved logging system with INFO/DEBUG level separation

✅ **Technical Stack:**
- LangGraph >= 0.2.0 for workflow orchestration
- LangChain Community for SQL database utilities
- Pydantic for data validation and structured outputs
- SQLAlchemy for database abstraction
- Python-dotenv for environment configuration
- SQLite database with Chinook sample data

✅ **Security Implementation:**
- Pattern-based SQL validation (blocks INSERT/UPDATE/DELETE/DROP/etc.)
- Query syntax checking with balanced parentheses validation
- Automatic LIMIT clause injection for result safety
- Read-only database operations enforced
- Input validation with length and content checks
- Dangerous pattern detection and blocking

## 5. Recent Fixes & Current Status

✅ **Recently Resolved Issues:**
- **GraphRecursionError**: Fixed infinite retry loops and improved workflow stop conditions
- **State Management**: Enhanced `StateManager.get_workflow_summary()` to handle different state formats
- **Trace Mode Response**: Fixed missing AI response display in trace mode execution
- **SQL Validation**: Reduced false positives in security validation patterns
- **Logging System**: Cleaned up INFO level logging and added comprehensive DEBUG logging
- **Workflow Visualization**: Added complete LangGraph structure display and execution tracking

🟡 **Medium Priority (Remaining):**
- **Type Compatibility**: `tuple[bool, str]` annotation in `utils/helpers.py:119` needs Python 3.9+
- **Database Path**: Potential mismatch between default config path and actual database location  
- **Path Resolution**: Complex database path logic may cause file not found errors in some environments

🟢 **Future Enhancements:**
- Add comprehensive test suite
- Enhance SQL injection detection patterns
- Add API documentation
- Implement query caching mechanism
- Add performance metrics and monitoring
- Add support for more database types
- Implement query optimization suggestions

## 6. New Debugging & Tracing Features

### 6.1 Workflow Visualization
- **Command**: `python main.py --show-graph`
- **Interactive**: `graph` command in interactive mode
- **Functionality**: Displays complete LangGraph workflow structure including:
  - All nodes and their connections
  - Conditional routing logic with decision functions
  - Retry mechanisms and error handling paths
  - Clear workflow execution flow description

### 6.2 Execution Tracing
- **Command**: `python main.py --trace --query "Your query"`
- **Functionality**: Real-time workflow execution tracking showing:
  - Step-by-step execution path through nodes
  - State changes and data flow between nodes
  - Retry attempts and error recovery
  - Complete execution history and timing

### 6.3 Enhanced State Management
- **Improvement**: `StateManager.get_workflow_summary()` now handles:
  - Both dict and TypedDict state formats
  - LangGraph streaming state accumulation
  - Pydantic object access patterns
  - Robust response data extraction for trace mode

### 6.4 Improved Logging System
- **INFO Level**: Clean, user-friendly progress messages
- **DEBUG Level**: Detailed technical information for troubleshooting
- **Structured Logging**: Clear node execution and routing decision logs

## 7. File Structure
```
├── core/                  # Core workflow logic
│   ├── state.py          # State management and data models
│   └── workflow.py       # LangGraph workflow implementation
├── security/             # Security validation
│   └── validator.py      # SQL safety checker
├── tools/               # Database utilities
│   └── database_tools.py # Database manager and query optimizer
├── config/              # Configuration
│   └── settings.py       # Pydantic-based settings
├── utils/               # Helper functions
│   └── helpers.py        # Logging, validation, formatting
├── data/                # Database files
│   └── Chinook.db       # Sample SQLite database
└── main.py              # Main application entry point
```