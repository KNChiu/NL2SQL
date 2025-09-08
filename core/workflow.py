from typing import Literal, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from langgraph.graph import StateGraph, START, END

from core.state import WorkflowState, WorkflowStage, StateManager
from security.validator import SQLValidator, QueryStatus


class NL2SQLWorkflow:
    """NL2SQL LangGraph Workflow Implementation"""
    
    def __init__(self, llm: ChatOpenAI, db: SQLDatabase):
        import logging
        self.logger = logging.getLogger(__name__)
        
        self.llm = llm
        self.db = db
        self.validator = SQLValidator()
        
        try:
            self.toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            self.tools = self.toolkit.get_tools()
            
            self.logger.debug(f"Available tools: {[tool.name for tool in self.tools]}")
            
            # Get specific tools
            self.list_tables_tool = next((tool for tool in self.tools if tool.name == "sql_db_list_tables"), None)
            self.get_schema_tool = next((tool for tool in self.tools if tool.name == "sql_db_schema"), None)
            
            if not self.list_tables_tool:
                self.logger.error("sql_db_list_tables tool not found")
                raise ValueError("Required tool 'sql_db_list_tables' not available")
                
            if not self.get_schema_tool:
                self.logger.error("sql_db_schema tool not found")
                raise ValueError("Required tool 'sql_db_schema' not available")
                
            self.logger.info("NL2SQL workflow tools initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize workflow tools: {e}")
            raise
        
        # Initialize workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        workflow = StateGraph(WorkflowState)
        
        # Add workflow nodes
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("retrieve_schema", self._retrieve_schema_node)
        workflow.add_node("generate_query", self._generate_query_node)
        workflow.add_node("validate_query", self._validate_query_node)
        workflow.add_node("execute_query", self._execute_query_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Define workflow edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "retrieve_schema")
        workflow.add_edge("retrieve_schema", "generate_query")
        
        # Add retry node to handle retry logic
        workflow.add_node("retry_increment", self._retry_increment_node)
        
        # Conditional edge from query generation
        workflow.add_conditional_edges(
            "generate_query",
            self._should_validate_or_retry,
            {
                "validate": "validate_query",
                "retry": "retry_increment",
                "error": "handle_error"
            }
        )
        
        # Conditional edge from validation
        workflow.add_conditional_edges(
            "validate_query", 
            self._should_execute_or_retry,
            {
                "execute": "execute_query",
                "retry": "retry_increment", 
                "error": "handle_error"
            }
        )
        
        # Conditional edge from execution
        workflow.add_conditional_edges(
            "execute_query",
            self._should_respond_or_retry,
            {
                "respond": "generate_response",
                "retry": "retry_increment",
                "error": "handle_error"
            }
        )
        
        # Always go back to generate_query after incrementing retry
        workflow.add_edge("retry_increment", "generate_query")
        
        workflow.add_edge("generate_response", END)
        workflow.add_edge("handle_error", END)
        
        return workflow
    
    def _initialize_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Initialize the workflow"""
        
        update = StateManager.update_stage(state, WorkflowStage.INITIALIZED)
        update["messages"] = [
            SystemMessage(content="Starting NL2SQL workflow for user query."),
            HumanMessage(content=state["user_query"])
        ]
        
        return update
    
    def _retrieve_schema_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Retrieve database schema information"""
        
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            logger.debug("Starting schema retrieval process")
            
            # Try primary method: Use LangChain tools
            try:
                # Get list of tables
                logger.debug("Invoking list_tables_tool")
                tables_result = self.list_tables_tool.invoke("")
                logger.debug(f"Raw tables result: {tables_result}")
                
                tables = self._parse_tables_from_result(tables_result)
                logger.debug(f"Parsed tables: {tables}")
                
            except Exception as tool_error:
                logger.warning(f"LangChain tools failed: {tool_error}")
                logger.debug("Trying fallback method: direct database access")
                
                # Fallback: Use database manager directly
                from tools.database_tools import db_manager
                tables = db_manager.list_tables()
                logger.debug(f"Fallback method got tables: {tables}")
            
            if not tables:
                logger.warning("No tables found in database")
                raise Exception("No tables found in database")
            
            # Select relevant tables based on user query
            selected_tables = self._select_relevant_tables(state["user_query"], tables)
            logger.debug(f"Selected relevant tables: {selected_tables}")
            
            if not selected_tables:
                logger.warning("No relevant tables selected")
                # Fallback: use first available table
                selected_tables = tables[:1]
                logger.debug(f"Using fallback table selection: {selected_tables}")
            
            # Get schema for selected tables
            schema_result = ""
            try:
                logger.debug(f"Getting schema for tables: {selected_tables}")
                # Convert list to comma-separated string for the tool
                table_names_str = ", ".join(selected_tables)
                schema_result = self.get_schema_tool.invoke({"table_names": table_names_str})
                logger.debug(f"Raw schema result: {schema_result}")
                
            except Exception as schema_error:
                logger.warning(f"Schema tool failed: {schema_error}")
                logger.debug("Trying fallback schema retrieval")
                
                # Fallback: Use database manager directly
                from tools.database_tools import db_manager
                schema_result = db_manager.get_table_schema(selected_tables)
                logger.debug(f"Fallback schema result: {schema_result}")
            
            if not schema_result:
                logger.warning("No schema information retrieved")
                schema_result = f"Tables available: {', '.join(selected_tables)}"
            
            updates = {}
            updates.update(StateManager.update_stage(state, WorkflowStage.SCHEMA_RETRIEVED))
            updates.update(StateManager.update_schema_info(
                state,
                tables=tables,
                selected_tables=selected_tables,
                table_schemas={"schema_text": schema_result}
            ))
            
            # Add schema info to messages
            schema_message = AIMessage(content=f"Retrieved schema for tables: {', '.join(selected_tables)}")
            updates["messages"] = state["messages"] + [schema_message]
            
            logger.debug("Schema retrieval completed successfully")
            return updates
            
        except Exception as e:
            logger.error(f"Schema retrieval failed: {str(e)}", exc_info=True)
            return StateManager.add_error(state, "SchemaRetrievalError", str(e))
    
    def _generate_query_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Generate SQL query from natural language"""
        
        try:
            # Create prompt for SQL generation
            sql_prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_sql_generation_prompt()),
                ("human", "User Query: {user_query}\n\nDatabase Schema: {schema}\n\nGenerate a SQL query:")
            ])
            
            schema_info = state.get("schema_info")
            schema_text = schema_info.table_schemas.get("schema_text", "")
            
            # Generate SQL query
            response = self.llm.invoke(
                sql_prompt.format_messages(
                    user_query=state["user_query"],
                    schema=schema_text
                )
            )
            
            # Extract SQL from response
            sql_query = self._extract_sql_from_response(response.content)
            
            updates = {}
            updates.update(StateManager.update_stage(state, WorkflowStage.QUERY_GENERATED))
            updates.update(StateManager.update_query_info(state, original_query=sql_query))
            
            # Add generation message
            gen_message = AIMessage(content=f"Generated SQL: {sql_query}")
            updates["messages"] = state["messages"] + [gen_message]
            
            return updates
            
        except Exception as e:
            return StateManager.add_error(state, "QueryGenerationError", str(e))
    
    def _validate_query_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Validate generated SQL query"""
        
        try:
            query_info = state.get("query_info")
            sql_query = query_info.original_query
            
            # Validate using security validator
            validation_result = self.validator.validate_query(sql_query)
            
            updates = {}
            updates.update(StateManager.update_stage(state, WorkflowStage.QUERY_VALIDATED))
            updates.update(StateManager.update_query_info(
                state,
                validation_result=validation_result,
                validated_query=sql_query if validation_result.is_safe else ""
            ))
            
            # Add validation message
            val_message = AIMessage(
                content=f"Validation result: {validation_result.status.value} - {validation_result.message}"
            )
            updates["messages"] = state["messages"] + [val_message]
            
            return updates
            
        except Exception as e:
            return StateManager.add_error(state, "QueryValidationError", str(e))
    
    def _execute_query_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute validated SQL query"""
        
        try:
            query_info = state.get("query_info")
            sql_query = query_info.validated_query
            
            # Execute query safely
            result = self.db.run_no_throw(sql_query)
            
            if not result:
                raise Exception("Query execution returned empty result")
            
            updates = {}
            updates.update(StateManager.update_stage(state, WorkflowStage.QUERY_EXECUTED))
            updates.update(StateManager.update_query_info(state, execution_result=result))
            
            # Add execution message
            exec_message = AIMessage(content=f"Query executed successfully. Result: {result}")
            updates["messages"] = state["messages"] + [exec_message]
            
            return updates
            
        except Exception as e:
            return StateManager.add_error(state, "QueryExecutionError", str(e))
    
    def _generate_response_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Generate natural language response"""
        
        try:
            query_info = state.get("query_info")
            execution_result = query_info.execution_result
            
            # Create prompt for response generation
            response_prompt = ChatPromptTemplate.from_messages([
                ("system", "Convert the SQL query results into a clear, natural language response for the user."),
                ("human", "User Query: {user_query}\nSQL Result: {result}\nProvide a natural language answer:")
            ])
            
            # Generate natural language response
            response = self.llm.invoke(
                response_prompt.format_messages(
                    user_query=state["user_query"],
                    result=execution_result
                )
            )
            
            updates = {}
            updates.update(StateManager.update_stage(state, WorkflowStage.RESPONSE_GENERATED))
            updates.update(StateManager.update_response_info(
                state,
                natural_language_response=response.content,
                structured_data={"sql_result": execution_result},
                confidence_score=0.9  # Could be calculated based on validation results
            ))
            
            # Add final response message
            response_message = AIMessage(content=response.content)
            updates["messages"] = state["messages"] + [response_message]
            
            return updates
            
        except Exception as e:
            return StateManager.add_error(state, "ResponseGenerationError", str(e))
    
    def _handle_error_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Handle workflow errors"""
        
        errors = state.get("errors", [])
        if not errors:
            return {"current_stage": WorkflowStage.ERROR}
        
        latest_error = errors[-1]
        
        # Generate error response
        error_response = f"I encountered an error while processing your query: {latest_error.message}"
        
        updates = StateManager.update_response_info(
            state,
            natural_language_response=error_response,
            confidence_score=0.0
        )
        
        error_message = AIMessage(content=error_response)
        updates["messages"] = state["messages"] + [error_message]
        
        return updates
    
    def _retry_increment_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Increment retry count and prepare for retry"""
        
        query_info = state.get("query_info")
        
        # Log retry attempt
        self.logger.info(f"Retrying workflow step (attempt {query_info.retry_count + 1}/{query_info.max_retries})")
        
        # Increment retry counter and reset query
        updates = StateManager.update_query_info(
            state,
            original_query="",  # Reset query for retry
            validated_query="",  # Reset validated query
            validation_result=None,  # Reset validation result
            execution_result=None,  # Reset execution result
            increment_retry=True
        )
        
        return updates
    
    def _should_validate_or_retry(self, state: WorkflowState) -> Literal["validate", "retry", "error"]:
        """Determine next step after query generation"""
        
        if state.get("current_stage") == WorkflowStage.ERROR:
            self.logger.debug("Current stage is ERROR, returning error")
            return "error"
            
        query_info = state.get("query_info")
        
        self.logger.debug(f"Query generation check - retry_count: {query_info.retry_count}/{query_info.max_retries}, "
                         f"original_query: {repr(query_info.original_query) if query_info else 'None'}")
        
        # Check if we've exceeded retry limits first
        if query_info.retry_count >= query_info.max_retries:
            self.logger.error(f"Maximum retries ({query_info.max_retries}) exceeded, stopping workflow")
            return "error"
        
        if not query_info or not query_info.original_query or query_info.original_query.strip() == "":
            if StateManager.should_retry(state):
                self.logger.warning(f"Query generation failed, retrying (attempt {query_info.retry_count + 1}/{query_info.max_retries})")
                return "retry"
            self.logger.error("Query generation failed after maximum retries")
            return "error"
        
        self.logger.debug("Query generation successful, proceeding to validation")
        return "validate"
    
    def _should_execute_or_retry(self, state: WorkflowState) -> Literal["execute", "retry", "error"]:
        """Determine next step after validation"""
        
        if state.get("current_stage") == WorkflowStage.ERROR:
            self.logger.debug("Current stage is ERROR, returning error")
            return "error"
            
        query_info = state.get("query_info")
        validation_result = query_info.validation_result
        
        self.logger.debug(f"Validation result: {validation_result.status.value if validation_result else 'None'}, "
                         f"is_safe: {validation_result.is_safe if validation_result else 'None'}, "
                         f"retry_count: {query_info.retry_count}/{query_info.max_retries}")
        
        # Check if we've exceeded retry limits first
        if query_info.retry_count >= query_info.max_retries:
            self.logger.error(f"Maximum retries ({query_info.max_retries}) exceeded, stopping workflow")
            return "error"
        
        if not validation_result or not validation_result.is_safe:
            if StateManager.should_retry(state):
                self.logger.warning(f"Query validation failed (safety), retrying (attempt {query_info.retry_count + 1}/{query_info.max_retries})")
                return "retry"
            self.logger.error("Query validation failed and cannot retry")
            return "error"
            
        if validation_result.status != QueryStatus.QUERY_IS_CORRECT:
            if StateManager.should_retry(state):
                self.logger.warning(f"Query validation failed (correctness), retrying (attempt {query_info.retry_count + 1}/{query_info.max_retries})")
                return "retry"
            self.logger.error("Query validation failed and cannot retry")
            return "error"
        
        self.logger.debug("Query validation passed, proceeding to execution")
        return "execute"
    
    def _should_respond_or_retry(self, state: WorkflowState) -> Literal["respond", "retry", "error"]:
        """Determine next step after execution"""
        
        if state.get("current_stage") == WorkflowStage.ERROR:
            self.logger.debug("Current stage is ERROR, returning error")
            return "error"
            
        query_info = state.get("query_info")
        
        self.logger.debug(f"Execution check - retry_count: {query_info.retry_count}/{query_info.max_retries}, "
                         f"execution_result exists: {bool(query_info.execution_result)}")
        
        # Check if we've exceeded retry limits first
        if query_info.retry_count >= query_info.max_retries:
            self.logger.error(f"Maximum retries ({query_info.max_retries}) exceeded, stopping workflow")
            return "error"
            
        if not query_info.execution_result:
            if StateManager.should_retry(state):
                self.logger.warning(f"Query execution failed, retrying (attempt {query_info.retry_count + 1}/{query_info.max_retries})")
                return "retry"
            self.logger.error("Query execution failed and cannot retry")
            return "error"
        
        self.logger.debug("Query execution successful, proceeding to response generation")
        return "respond"
    
    def _parse_tables_from_result(self, tables_result: str) -> list:
        """Parse table names from tool result"""
        import re
        
        self.logger.info(f"Parsing tables from result: {repr(tables_result)}")
        
        if not tables_result:
            self.logger.warning("Empty tables result received")
            return []
            
        tables = []
        
        try:
            # Handle different possible formats from the tool
            if isinstance(tables_result, str):
                # Remove common prefixes/suffixes
                cleaned_result = tables_result.strip()
                
                # Try to extract table names from various formats
                # Format 1: Simple comma-separated list
                if ',' in cleaned_result and not '\n' in cleaned_result:
                    tables = [t.strip() for t in cleaned_result.split(',') if t.strip()]
                
                # Format 2: Line-separated list
                elif '\n' in cleaned_result:
                    lines = cleaned_result.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('--') and not line.startswith('#'):
                            # Extract table name from potential formatted line
                            # Handle cases like "Table: users" or "- users" or just "users"
                            match = re.search(r'(?:Table:\s*|[-*]\s*)?(\w+)', line)
                            if match:
                                tables.append(match.group(1))
                            elif line.isalpha() or '_' in line:
                                tables.append(line)
                
                # Format 3: Single table or simple format
                else:
                    # Try to extract valid table names using regex
                    table_matches = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', cleaned_result)
                    if table_matches:
                        tables = table_matches
                    elif cleaned_result.replace('_', '').replace(' ', '').isalnum():
                        tables = [cleaned_result]
            
            # Remove duplicates while preserving order
            tables = list(dict.fromkeys(tables))
            
            self.logger.debug(f"Successfully parsed {len(tables)} tables: {tables}")
            return tables
            
        except Exception as e:
            self.logger.error(f"Error parsing tables: {e}")
            self.logger.error(f"Raw result was: {repr(tables_result)}")
            return []
    
    def _select_relevant_tables(self, user_query: str, available_tables: list) -> list:
        """Select relevant tables based on user query - simplified version"""
        # For now, return all tables. In production, use LLM to select relevant ones
        return available_tables[:5]  # Limit to first 5 tables for performance
    
    def _extract_sql_from_response(self, response_content: str) -> str:
        """Extract SQL query from LLM response"""
        import re
        
        # Look for SQL in code blocks
        sql_match = re.search(r'```sql\n(.*?)\n```', response_content, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        
        # Look for SQL in generic code blocks
        code_match = re.search(r'```\n(.*?)\n```', response_content, re.DOTALL)
        if code_match:
            content = code_match.group(1).strip()
            if content.upper().startswith('SELECT'):
                return content
        
        # Look for SELECT statements in plain text
        select_match = re.search(r'(SELECT\b.*?)(?:\n\n|\Z)', response_content, re.DOTALL | re.IGNORECASE)
        if select_match:
            return select_match.group(1).strip()
        
        return response_content.strip()
    
    def _get_sql_generation_prompt(self) -> str:
        """Get system prompt for SQL generation"""
        
        return """You are a SQL expert. Generate syntactically correct SQLite queries based on user requests.

Rules:
1. Only generate SELECT queries - no INSERT, UPDATE, DELETE, DROP, CREATE, etc.
2. Use proper SQLite syntax
3. Include appropriate WHERE clauses for filtering
4. Use LIMIT 5 unless user specifies otherwise
5. Return only the SQL query without explanations

Generate a clean SQL query that answers the user's question."""
    
    def compile(self):
        """Compile the workflow"""
        return self.workflow.compile()