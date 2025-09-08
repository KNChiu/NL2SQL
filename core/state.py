from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

from langgraph.graph.message import AnyMessage, add_messages
from security.validator import QueryStatus, ValidationResult


class WorkflowStage(str, Enum):
    """Stages in the NL2SQL workflow"""
    INITIALIZED = "INITIALIZED"
    SCHEMA_RETRIEVED = "SCHEMA_RETRIEVED" 
    QUERY_GENERATED = "QUERY_GENERATED"
    QUERY_VALIDATED = "QUERY_VALIDATED"
    QUERY_EXECUTED = "QUERY_EXECUTED"
    RESPONSE_GENERATED = "RESPONSE_GENERATED"
    ERROR = "ERROR"


class ErrorInfo(BaseModel):
    """Error information structure"""
    stage: WorkflowStage
    error_type: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


class QueryInfo(BaseModel):
    """SQL query information"""
    original_query: str = ""
    validated_query: str = ""
    validation_result: Optional[ValidationResult] = None
    execution_result: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class SchemaInfo(BaseModel):
    """Database schema information"""
    tables: List[str] = Field(default_factory=list)
    selected_tables: List[str] = Field(default_factory=list)
    table_schemas: Dict[str, Any] = Field(default_factory=dict)


class ResponseInfo(BaseModel):
    """Final response information"""
    natural_language_response: str = ""
    structured_data: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = 0.0


class WorkflowState(TypedDict):
    """
    LangGraph state for NL2SQL workflow
    
    Attributes:
        messages: LangGraph messages for LLM communication
        user_query: Original natural language query from user
        current_stage: Current workflow stage
        schema_info: Database schema information
        query_info: SQL query generation and validation info
        response_info: Final response information
        errors: List of errors encountered during workflow
        metadata: Additional metadata for the workflow
    """
    messages: Annotated[List[AnyMessage], add_messages]
    user_query: str
    current_stage: WorkflowStage
    schema_info: SchemaInfo
    query_info: QueryInfo
    response_info: ResponseInfo
    errors: List[ErrorInfo]
    metadata: Dict[str, Any]


class StateManager:
    """Utility class for managing workflow state"""
    
    @staticmethod
    def create_initial_state(user_query: str) -> Dict[str, Any]:
        """Create initial workflow state"""
        return {
            "messages": [],
            "user_query": user_query,
            "current_stage": WorkflowStage.INITIALIZED,
            "schema_info": SchemaInfo(),
            "query_info": QueryInfo(),
            "response_info": ResponseInfo(),
            "errors": [],
            "metadata": {
                "start_time": datetime.now(),
                "workflow_id": f"nl2sql_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
        }
    
    @staticmethod
    def update_stage(state: WorkflowState, new_stage: WorkflowStage) -> Dict[str, Any]:
        """Update workflow stage"""
        return {"current_stage": new_stage}
    
    @staticmethod
    def add_error(state: WorkflowState, error_type: str, message: str) -> Dict[str, Any]:
        """Add error to state"""
        error = ErrorInfo(
            stage=state["current_stage"],
            error_type=error_type,
            message=message
        )
        
        errors = state.get("errors", [])
        errors.append(error)
        
        return {
            "errors": errors,
            "current_stage": WorkflowStage.ERROR
        }
    
    @staticmethod
    def update_schema_info(
        state: WorkflowState, 
        tables: Optional[List[str]] = None,
        selected_tables: Optional[List[str]] = None,
        table_schemas: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update schema information"""
        
        current_schema = state.get("schema_info", SchemaInfo())
        
        updated_schema = SchemaInfo(
            tables=tables if tables is not None else current_schema.tables,
            selected_tables=selected_tables if selected_tables is not None else current_schema.selected_tables,
            table_schemas=table_schemas if table_schemas is not None else current_schema.table_schemas
        )
        
        return {"schema_info": updated_schema}
    
    @staticmethod
    def update_query_info(
        state: WorkflowState,
        original_query: Optional[str] = None,
        validated_query: Optional[str] = None,
        validation_result: Optional[ValidationResult] = None,
        execution_result: Optional[str] = None,
        increment_retry: bool = False
    ) -> Dict[str, Any]:
        """Update query information"""
        
        current_query = state.get("query_info", QueryInfo())
        
        retry_count = current_query.retry_count
        if increment_retry:
            retry_count += 1
            
        updated_query = QueryInfo(
            original_query=original_query if original_query is not None else current_query.original_query,
            validated_query=validated_query if validated_query is not None else current_query.validated_query,
            validation_result=validation_result if validation_result is not None else current_query.validation_result,
            execution_result=execution_result if execution_result is not None else current_query.execution_result,
            retry_count=retry_count,
            max_retries=current_query.max_retries
        )
        
        return {"query_info": updated_query}
    
    @staticmethod
    def update_response_info(
        state: WorkflowState,
        natural_language_response: Optional[str] = None,
        structured_data: Optional[Dict[str, Any]] = None,
        confidence_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """Update response information"""
        
        current_response = state.get("response_info", ResponseInfo())
        
        updated_response = ResponseInfo(
            natural_language_response=natural_language_response if natural_language_response is not None else current_response.natural_language_response,
            structured_data=structured_data if structured_data is not None else current_response.structured_data,
            confidence_score=confidence_score if confidence_score is not None else current_response.confidence_score
        )
        
        return {"response_info": updated_response}
    
    @staticmethod
    def should_retry(state: WorkflowState) -> bool:
        """Check if workflow should retry based on error count"""
        query_info = state.get("query_info", QueryInfo())
        return query_info.retry_count < query_info.max_retries
    
    @staticmethod
    def get_workflow_summary(state: WorkflowState) -> Dict[str, Any]:
        """Get structured summary of workflow state for JSON output"""
        import logging
        logger = logging.getLogger(__name__)
        
        # Handle both dict and TypedDict access patterns
        if isinstance(state, dict):
            metadata = state.get("metadata", {})
            user_query = state.get("user_query", "")
            current_stage = state.get("current_stage", WorkflowStage.INITIALIZED)
            query_info = state.get("query_info", QueryInfo())
            response_info = state.get("response_info", ResponseInfo())
            errors = state.get("errors", [])
        else:
            # Handle Pydantic object access
            metadata = getattr(state, "metadata", {})
            user_query = getattr(state, "user_query", "")
            current_stage = getattr(state, "current_stage", WorkflowStage.INITIALIZED)
            query_info = getattr(state, "query_info", QueryInfo())
            response_info = getattr(state, "response_info", ResponseInfo())
            errors = getattr(state, "errors", [])
        
        # Debug logging for response_info
        logger.debug(f"Response info type: {type(response_info)}")
        logger.debug(f"Response info content: {response_info}")
        
        # Extract natural language response safely
        natural_language = ""
        structured_data = {}
        confidence = 0.0
        
        if hasattr(response_info, 'natural_language_response'):
            natural_language = response_info.natural_language_response
            structured_data = response_info.structured_data
            confidence = response_info.confidence_score
        elif isinstance(response_info, dict):
            natural_language = response_info.get("natural_language_response", "")
            structured_data = response_info.get("structured_data", {})
            confidence = response_info.get("confidence_score", 0.0)
        
        logger.debug(f"Extracted natural_language: {repr(natural_language)}")
        
        # Extract SQL query info safely
        sql_query = ""
        validation_status = None
        retry_count = 0
        
        if hasattr(query_info, 'validated_query'):
            sql_query = query_info.validated_query
            validation_status = query_info.validation_result.status.value if query_info.validation_result else None
            retry_count = query_info.retry_count
        elif isinstance(query_info, dict):
            sql_query = query_info.get("validated_query", "")
            val_result = query_info.get("validation_result")
            validation_status = val_result.status.value if val_result else None
            retry_count = query_info.get("retry_count", 0)
        
        # Extract current stage value
        stage_value = current_stage
        if hasattr(current_stage, 'value'):
            stage_value = current_stage.value
        
        return {
            "workflow_id": metadata.get("workflow_id", "unknown") if isinstance(metadata, dict) else "unknown",
            "user_query": user_query,
            "current_stage": stage_value,
            "status": "success" if stage_value == WorkflowStage.RESPONSE_GENERATED else "in_progress",
            "query_info": {
                "sql_query": sql_query,
                "validation_status": validation_status,
                "retry_count": retry_count
            },
            "response": {
                "natural_language": natural_language,
                "structured_data": structured_data,
                "confidence": confidence
            },
            "errors": [
                {
                    "stage": error.stage.value,
                    "type": error.error_type,
                    "message": error.message,
                    "timestamp": error.timestamp.isoformat()
                } for error in errors
            ],
            "execution_time": (
                datetime.now() - metadata.get("start_time", datetime.now())
            ).total_seconds() if isinstance(metadata, dict) else 0.0
        }