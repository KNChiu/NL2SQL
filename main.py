#!/usr/bin/env python3
"""
NL2SQL LangGraph Workflow - Main Entry Point

A modular, secure, and extensible Natural Language to SQL system
built with LangGraph for workflow orchestration.
"""

import sys
import json
import argparse
import time
import signal
import atexit
from pathlib import Path
from typing import Dict, Any

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase

from config.settings import settings
from core.workflow import NL2SQLWorkflow
from core.state import StateManager, WorkflowState
from tools.database_tools import db_manager
from utils.helpers import (
    setup_logging, 
    validate_user_input, 
    save_workflow_result,
    create_success_response,
    create_error_response,
    format_execution_time,
    log_function_call,
    get_user_friendly_error,
    create_debug_info
)
from security.validator import SQLValidator


class NL2SQLApp:
    """Main NL2SQL application"""
    
    def __init__(self):
        self.logger = setup_logging(settings.workflow.enable_debug)
        self.llm = None
        self.db = None
        self.workflow = None
        self.validator = SQLValidator()
        self._shutdown = False
        
        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def initialize(self) -> bool:
        """Initialize the application components"""
        
        try:
            # Validate configuration
            settings.validate_config()
            
            # Initialize LLM
            self.llm = ChatOpenAI(
                model=settings.llm.model,
                base_url=settings.llm.base_url,
                api_key=settings.llm.api_key,
                temperature=settings.llm.temperature,
                max_tokens=settings.llm.max_tokens,
                seed=settings.llm.seed
            )
            
            # Initialize database
            self.db = db_manager.get_database()
            
            # Test database connection
            self.logger.debug("Testing database connection...")
            if not db_manager.test_connection():
                self.logger.error("Database connection test failed")
                raise Exception("Failed to connect to database")
            
            self.logger.debug("Database connection successful")
            
            # Initialize workflow
            self.workflow = NL2SQLWorkflow(self.llm, self.db)
            self.compiled_workflow = self.workflow.compile()
            
            # Set recursion limit to prevent infinite loops
            # Use configured recursion limit to accommodate retry logic while still preventing infinite loops
            self.recursion_limit = settings.workflow.recursion_limit
            if hasattr(self.compiled_workflow, 'config'):
                self.compiled_workflow.config = {"recursion_limit": self.recursion_limit}
            else:
                # For newer versions of LangGraph, set via invoke parameter
                pass
            
            self.logger.info("NL2SQL application initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize application: {e}")
            return False
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Process a natural language query through the workflow"""
        
        start_time = time.time()
        
        try:
            # Validate user input
            is_valid, validation_message = validate_user_input(user_query)
            if not is_valid:
                return create_error_response(validation_message, "InputValidationError")
            
            # Create initial state
            initial_state = StateManager.create_initial_state(user_query)
            
            # Execute workflow with recursion limit
            self.logger.debug(f"Processing query: {user_query}")
            result_state = self.compiled_workflow.invoke(
                initial_state,
                config={"recursion_limit": self.recursion_limit}
            )
            
            # Get workflow summary
            summary = StateManager.get_workflow_summary(result_state)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            summary["execution_time_formatted"] = format_execution_time(execution_time)
            
            # Save result if enabled
            if settings.workflow.enable_tracing:
                save_workflow_result(summary["workflow_id"], summary)
            
            # Return success response
            return create_success_response(summary, "Query processed successfully")
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log detailed error information
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            log_function_call("process_query", {"user_query": user_query[:100]}, self.logger)
            
            # Special handling for GraphRecursionError
            from langgraph.errors import GraphRecursionError
            if isinstance(e, GraphRecursionError):
                self.logger.error("GraphRecursionError detected - workflow exceeded recursion limit")
                friendly_message = "🔄 Workflow execution error: The system encountered too many retry attempts. This might indicate a complex query or validation issue."
                error_type = "GraphRecursionError"
                suggestions = [
                    "Try rephrasing your query in simpler terms",
                    "Check if you're asking about data that exists in the database",
                    "Use 'info' command to see available tables",
                    "Try enabling debug mode with --debug for more details"
                ]
            else:
                # Create user-friendly error message for other errors
                friendly_message = get_user_friendly_error(e)
                error_type = "WorkflowExecutionError"
                suggestions = [
                    "Try rephrasing your query in simpler terms",
                    "Check if you're asking about data that exists in the database", 
                    "Use 'info' command to see available tables"
                ]
            
            # Create debug info for troubleshooting
            debug_info = create_debug_info()
            debug_info["execution_time"] = execution_time
            debug_info["original_error"] = str(e)
            debug_info["recursion_limit"] = self.recursion_limit
            
            return create_error_response(
                friendly_message, 
                error_type,
                details=debug_info if settings.workflow.enable_debug else None,
                suggestions=suggestions
            )
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information"""
        
        try:
            db_info = db_manager.get_database_info()
            return create_success_response(db_info, "Database info retrieved")
        except Exception as e:
            return create_error_response(f"Failed to get database info: {e}")
    
    def validate_sql(self, sql_query: str) -> Dict[str, Any]:
        """Validate SQL query independently"""
        
        try:
            validation_result = self.validator.validate_query(sql_query)
            
            result = {
                "status": validation_result.status.value,
                "is_safe": validation_result.is_safe,
                "message": validation_result.message,
                "corrected_query": validation_result.corrected_query
            }
            
            if not validation_result.is_safe:
                suggestions = self.validator.get_safe_query_suggestions(sql_query)
                result["suggestions"] = suggestions
            
            return create_success_response(result, "SQL validation completed")
            
        except Exception as e:
            return create_error_response(f"SQL validation failed: {e}")
    
    def interactive_mode(self):
        """Run interactive query mode"""
        
        print("🚀 NL2SQL Interactive Mode")
        print("Enter your natural language queries. Type 'exit' to quit.")
        print("Type 'info' to see database information.")
        print("Type 'help' for available commands.\n")
        
        while not self._shutdown:
            try:
                user_input = input("💬 Query: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                if user_input.lower() == 'graph':
                    self.show_workflow_graph()
                    continue
                
                if user_input.lower() == 'info':
                    result = self.get_database_info()
                    self._print_result(result)
                    continue
                
                if not user_input:
                    continue
                
                # Check for shutdown before processing
                if self._shutdown:
                    break
                
                # Process the query
                result = self.process_query(user_input)
                self._print_result(result)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                self._shutdown = True
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_help(self):
        """Show available commands"""
        print("\n📋 Available Commands:")
        print("  help  - Show this help message")
        print("  info  - Show database information")
        print("  graph - Show workflow graph structure")
        print("  exit  - Exit the application")
        print("  Any other text will be processed as a natural language query\n")
        print("📝 CLI Options:")
        print("  --trace         - Enable execution tracing")
        print("  --show-graph    - Display workflow structure and exit")
        print("  --debug         - Enable debug logging")
        print("  --db-info       - Show database info and exit\n")
    
    def _print_result(self, result: Dict[str, Any]):
        """Print formatted result"""
        
        if result["success"]:
            data = result["data"]
            
            if "natural_language" in data.get("response", {}):
                # Display AI response in a formatted block
                response_text = data['response']['natural_language']
                print("\n" + "="*60)
                print("🤖 AI Response")
                print("="*60)
                print(response_text)
                print("="*60)
                
                # Show SQL query if available
                sql_query = data.get("query_info", {}).get("sql_query", "")
                if sql_query:
                    print(f"\n🔍 Generated SQL:")
                    print("-"*40)
                    print(sql_query)
                    print("-"*40)
                
                # Show structured data if available
                structured_data = data.get("response", {}).get("structured_data", {})
                if structured_data.get("sql_result"):
                    print(f"\n📊 Raw Result:")
                    print("-"*40)
                    print(structured_data['sql_result'])
                    print("-"*40)
            else:
                # For other types of results (like database info)
                print("\n" + "="*60)
                print("📋 System")
                print("="*60)
                print(json.dumps(data, indent=2, ensure_ascii=False))
                print("="*60)
        else:
            error = result["error"]
            print(f"\n{error['message']}")
            
            # Show error ID for support reference
            if "error_id" in error:
                print(f"Error ID: {error['error_id']}")
            
            # Show suggestions if available
            if "suggestions" in error and error["suggestions"]:
                print("\n💡 Suggestions:")
                for i, suggestion in enumerate(error["suggestions"], 1):
                    print(f"  {i}. {suggestion}")
            
            # Show debug details if available and enabled
            if "details" in error and settings.workflow.enable_debug:
                print(f"\n🔧 Debug Info:")
                details = error["details"]
                if "execution_time" in details:
                    print(f"  Execution time: {format_execution_time(details['execution_time'])}")
                if "original_error" in details:
                    print(f"  Technical error: {details['original_error']}")
                if "workflow_state" in details:
                    ws = details["workflow_state"]
                    print(f"  Workflow stage: {ws.get('current_stage', 'Unknown')}")
                    print(f"  Retry count: {ws.get('retry_count', 0)}")
                    print(f"  Errors: {ws.get('errors_count', 0)}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown = True
        print("\n📴 Shutting down gracefully...")
        self.cleanup()
        sys.exit(0)
    
    def show_workflow_graph(self):
        """Display the workflow graph structure"""
        if not self.workflow:
            print("❌ Workflow not initialized")
            return
        
        self.workflow.print_graph_structure()
    
    def process_query_with_trace(self, user_query: str) -> Dict[str, Any]:
        """Process query with execution tracing enabled"""
        
        start_time = time.time()
        state_history = []
        
        try:
            # Validate user input
            is_valid, validation_message = validate_user_input(user_query)
            if not is_valid:
                return create_error_response(validation_message, "InputValidationError")
            
            # Create initial state
            initial_state = StateManager.create_initial_state(user_query)
            state_history.append(initial_state.copy())
            
            # Execute workflow with tracing
            self.logger.info(f"Processing query with tracing: {user_query}")
            
            # Use stream to capture intermediate states and build final state
            accumulated_state = initial_state.copy()
            final_state = None
            
            for i, step in enumerate(self.compiled_workflow.stream(
                initial_state,
                config={"recursion_limit": self.recursion_limit}
            )):
                self.logger.debug(f"Stream step {i}: type={type(step)}, keys={list(step.keys()) if isinstance(step, dict) else 'N/A'}")
                self.logger.debug(f"Stream step {i} content: {step}")
                
                # Accumulate state changes from each node
                if isinstance(step, dict):
                    for node_name, node_result in step.items():
                        if isinstance(node_result, dict):
                            # Update accumulated state with node result
                            accumulated_state.update(node_result)
                            self.logger.debug(f"Updated accumulated state with {node_name} result")
                
                state_history.append(step.copy() if hasattr(step, 'copy') else step)
                final_state = step
            
            if final_state is None:
                raise Exception("Workflow did not produce any output")
            
            # Print execution trace
            self.workflow.print_execution_path(state_history)
            
            # Debug final accumulated state
            self.logger.debug(f"Final accumulated state type: {type(accumulated_state)}")
            self.logger.debug(f"Final accumulated state keys: {list(accumulated_state.keys()) if isinstance(accumulated_state, dict) else 'N/A'}")
            self.logger.debug(f"Accumulated state response_info: {accumulated_state.get('response_info', 'NOT_FOUND')}")
            
            # Get workflow summary from accumulated state
            summary = StateManager.get_workflow_summary(accumulated_state)
            
            # Debug summary content
            self.logger.debug(f"Workflow summary: {summary}")
            
            # Calculate execution time
            execution_time = time.time() - start_time
            summary["execution_time_formatted"] = format_execution_time(execution_time)
            summary["execution_trace"] = state_history
            
            # Save result if enabled
            if settings.workflow.enable_tracing:
                save_workflow_result(summary["workflow_id"], summary)
            
            # Return success response
            return create_success_response(summary, "Query processed successfully with trace")
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log detailed error information
            self.logger.error(f"Error processing query with trace: {e}", exc_info=True)
            
            # Print partial execution trace if available
            if state_history:
                print("\n" + "="*60)
                print("PARTIAL EXECUTION TRACE (before error)")
                print("="*60)
                self.workflow.print_execution_path(state_history)
            
            # Handle error similar to regular process_query
            from langgraph.errors import GraphRecursionError
            if isinstance(e, GraphRecursionError):
                friendly_message = "Workflow execution error: The system encountered too many retry attempts with tracing enabled."
                error_type = "GraphRecursionError"
            else:
                friendly_message = get_user_friendly_error(e)
                error_type = "WorkflowExecutionError"
            
            debug_info = create_debug_info()
            debug_info["execution_time"] = execution_time
            debug_info["original_error"] = str(e)
            debug_info["recursion_limit"] = self.recursion_limit
            debug_info["state_history"] = state_history
            
            return create_error_response(
                friendly_message,
                error_type,
                details=debug_info if settings.workflow.enable_debug else None,
                suggestions=[
                    "Try rephrasing your query in simpler terms",
                    "Check if you're asking about data that exists in the database",
                    "Use 'info' command to see available tables"
                ]
            )
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'logger') and self.logger:
            self.logger.info("Cleaning up application resources...")
        
        try:
            # Close database connections if they exist
            if hasattr(self, 'db') and self.db and hasattr(self.db, '_engine'):
                self.db._engine.dispose()
                if hasattr(self, 'logger') and self.logger:
                    self.logger.debug("Database connections closed")
            
            # Clean up old result files (keep last 7 days)
            from utils.helpers import clean_old_results
            deleted_count = clean_old_results(days_old=7)
            if hasattr(self, 'logger') and self.logger:
                self.logger.debug(f"Cleaned up {deleted_count} old result files")
        
        except Exception as e:
            if hasattr(self, 'logger') and self.logger:
                self.logger.error(f"Error during cleanup: {e}")
            else:
                print(f"Error during cleanup: {e}")


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="NL2SQL LangGraph Workflow")
    parser.add_argument(
        "--query", 
        type=str, 
        help="Natural language query to process"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--validate-sql",
        type=str,
        help="Validate a SQL query"
    )
    parser.add_argument(
        "--db-info",
        action="store_true", 
        help="Show database information"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="Display the workflow graph structure"
    )
    parser.add_argument(
        "--trace",
        action="store_true", 
        help="Enable execution tracing to see workflow path"
    )
    
    args = parser.parse_args()
    
    # Override debug setting if specified
    if args.debug:
        settings.workflow.enable_debug = True
    
    # Initialize application
    app = NL2SQLApp()
    
    if not app.initialize():
        print("❌ Failed to initialize NL2SQL application")
        sys.exit(1)
    
    # Handle different modes
    if args.show_graph:
        app.show_workflow_graph()
    elif args.interactive:
        app.interactive_mode()
    elif args.query:
        if args.trace:
            result = app.process_query_with_trace(args.query)
        else:
            result = app.process_query(args.query)
        app._print_result(result)
    elif args.validate_sql:
        result = app.validate_sql(args.validate_sql)
        app._print_result(result)
    elif args.db_info:
        result = app.get_database_info()
        app._print_result(result)
    else:
        # Default to interactive mode
        app.interactive_mode()


if __name__ == "__main__":
    main()