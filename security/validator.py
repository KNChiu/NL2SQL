import re
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from pydantic import BaseModel


class QueryStatus(str, Enum):
    """SQL query validation status"""
    QUERY_IS_CORRECT = "QUERY_IS_CORRECT"
    QUERY_IS_NOT_CORRECT = "QUERY_IS_NOT_CORRECT"
    QUERY_IS_UNSAFE = "QUERY_IS_UNSAFE"


class ValidationResult(BaseModel):
    """Result of SQL validation"""
    status: QueryStatus
    message: str
    is_safe: bool
    corrected_query: str = ""


class SQLValidator:
    """SQL security and correctness validator"""
    
    # Dangerous SQL patterns that should be blocked
    DANGEROUS_PATTERNS = [
        # DML operations
        r'\bINSERT\b',
        r'\bUPDATE\b', 
        r'\bDELETE\b',
        r'\bREPLACE\b',
        r'\bMERGE\b',
        # DDL operations
        r'\bDROP\b',
        r'\bCREATE\b',
        r'\bALTER\b',
        r'\bTRUNCATE\b',
        r'\bRENAME\b',
        # DCL operations
        r'\bGRANT\b',
        r'\bREVOKE\b',
        # System functions and commands
        r'\bEXEC\b',
        r'\bEXECUTE\b',
        r'\bSYSTEM\b',
        r'\bSHELL\b',
        r'\bCMD\b',
        r'\bXP_CMDSHELL\b',
        # File operations
        r'\bLOAD_FILE\b',
        r'\bINTO\s+OUTFILE\b',
        r'\bINTO\s+DUMPFILE\b',
        r'\bSELECT\s+.*\s+INTO\s+OUTFILE\b',
        # Database info functions (potentially risky)
        r'\bUSER\(\)',
        r'\bDATABASE\(\)',
        r'\bVERSION\(\)',
        r'\bCURRENT_USER\b',
        r'\bSESSION_USER\b',
        # Union-based injections (only clearly malicious patterns)
        r'\bUNION\s+(?:ALL\s+)?SELECT\b.*\b(OR|AND)\s+\d+\s*=\s*\d+',
        # Comment-based injections
        r'--\s*$',
        r'/\*.*\*/',
        r'#.*$',
        # Time-based attack patterns
        r'\bSLEEP\s*\(',
        r'\bWAIT\s+FOR\s+DELAY\b',
        r'\bPG_SLEEP\s*\(',
        # Boolean-based injections (only obvious injection patterns)
        r'\bOR\s+1\s*=\s*1\b',
        r'\bAND\s+1\s*=\s*1\b',
        r'\bOR\s+\'[^\']*\'\s*=\s*\'[^\']*\'\s*--',
        # Subquery injections
        r'\(\s*SELECT\b(?![^)]*\bFROM\s+\w+\s*\))',
    ]
    
    # Common SQL errors to check
    COMMON_ERRORS = [
        (r'\bNOT\s+IN\s*\([^)]*NULL[^)]*\)', "Using NOT IN with NULL values"),
        (r'\bUNION\b(?!\s+ALL)', "Consider using UNION ALL instead of UNION"),
        (r'\bBETWEEN\s+\S+\s+AND\s+\S+', "Check if BETWEEN range is correct"),
        (r'SELECT\s+\*\s+FROM\s+\w+\s*$', "Consider specifying column names instead of SELECT *"),
    ]
    
    # Additional suspicious patterns that might indicate injection attempts
    # Made more specific to reduce false positives
    SUSPICIOUS_PATTERNS = [
        (r'[\'"]\s*;\s*--', "Potential SQL injection: quote followed by semicolon and comment"),
        (r'[\'"]\s*\|\|', "Potential SQL injection: quote followed by concatenation"),
        (r'\b\w+\s*=\s*\w+\s*--', "Potential SQL injection: comparison followed by comment"),
        # Removed overly broad patterns that cause false positives:
        # - Multiple quoted strings (common in legitimate queries)
        # - Numeric comparisons (very common in WHERE clauses)
        # - String concatenation (can be legitimate)
        
        # Keep only clearly malicious patterns
        (r'\'.*\'.*\'.*\'', "Excessive quoted strings might indicate injection attempt"),
        (r'\b(OR|AND)\s+[\'"]?\w+[\'"]?\s*=\s*[\'"]?\w+[\'"]?\s*--', "Condition with comment might be injection"),
    ]
    
    def __init__(self):
        self.dangerous_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.DANGEROUS_PATTERNS]
        self.error_regex = [(re.compile(pattern, re.IGNORECASE), message) for pattern, message in self.COMMON_ERRORS]
        self.suspicious_regex = [(re.compile(pattern, re.IGNORECASE), message) for pattern, message in self.SUSPICIOUS_PATTERNS]
    
    def validate_query(self, query: str) -> ValidationResult:
        """
        Validate SQL query for safety and correctness
        
        Args:
            query: SQL query string to validate
            
        Returns:
            ValidationResult with status and recommendations
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Starting validation for query: {repr(query)}")
        
        if not query or not query.strip():
            logger.debug("Query is empty or None")
            return ValidationResult(
                status=QueryStatus.QUERY_IS_NOT_CORRECT,
                message="Empty query provided",
                is_safe=False
            )
        
        # Clean query for analysis
        clean_query = query.strip()
        logger.debug(f"Clean query: {repr(clean_query)}")
        
        # Check for dangerous operations
        logger.debug("Performing safety check...")
        safety_check = self._check_safety(clean_query)
        logger.debug(f"Safety check result: {safety_check.status.value}, safe: {safety_check.is_safe}")
        if not safety_check.is_safe:
            logger.warning(f"Safety check failed: {safety_check.message}")
            return safety_check
        
        # Check for SQL injection attempts
        logger.debug("Performing injection pattern check...")
        injection_check = self._check_injection_patterns(clean_query)
        logger.debug(f"Injection check result: {injection_check.status.value}, safe: {injection_check.is_safe}")
        if not injection_check.is_safe:
            logger.warning(f"Injection check failed: {injection_check.message}")
            return injection_check
            
        # Check for common SQL errors
        logger.debug("Performing correctness check...")
        correctness_check = self._check_correctness(clean_query)
        logger.debug(f"Correctness check result: {correctness_check.status.value}, safe: {correctness_check.is_safe}")
        
        logger.info(f"Query validation completed with status: {correctness_check.status.value}")
        return correctness_check
    
    def _check_safety(self, query: str) -> ValidationResult:
        """Check if query contains dangerous operations"""
        
        # Must be SELECT only for safety
        if not re.match(r'^\s*SELECT\b', query, re.IGNORECASE):
            return ValidationResult(
                status=QueryStatus.QUERY_IS_UNSAFE,
                message="Only SELECT queries are allowed",
                is_safe=False
            )
        
        # Check for dangerous patterns
        for pattern in self.dangerous_regex:
            if pattern.search(query):
                return ValidationResult(
                    status=QueryStatus.QUERY_IS_UNSAFE,
                    message=f"Dangerous operation detected: {pattern.pattern}",
                    is_safe=False
                )
        
        return ValidationResult(
            status=QueryStatus.QUERY_IS_CORRECT,
            message="Query passed safety checks",
            is_safe=True
        )
    
    def _check_injection_patterns(self, query: str) -> ValidationResult:
        """Check for SQL injection patterns"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.debug(f"Checking injection patterns for query: {repr(query)}")
        suspicious_issues = []
        
        # Check for suspicious injection patterns
        logger.debug("Checking suspicious regex patterns...")
        for pattern, message in self.suspicious_regex:
            if pattern.search(query):
                logger.warning(f"Suspicious pattern matched: {pattern.pattern} -> {message}")
                suspicious_issues.append(message)
        
        # Additional heuristic checks
        logger.debug("Checking query structure...")
        structure_suspicious, structure_reasons = self._has_suspicious_structure_detailed(query)
        if structure_suspicious:
            logger.warning(f"Suspicious structure detected: {structure_reasons}")
            suspicious_issues.extend(structure_reasons)
        
        if suspicious_issues:
            logger.warning(f"Injection patterns detected: {suspicious_issues}")
            return ValidationResult(
                status=QueryStatus.QUERY_IS_UNSAFE,
                message=f"Potential SQL injection detected: {'; '.join(suspicious_issues)}",
                is_safe=False
            )
        
        logger.debug("No injection patterns detected")
        return ValidationResult(
            status=QueryStatus.QUERY_IS_CORRECT,
            message="No injection patterns detected",
            is_safe=True
        )
    
    def _has_suspicious_structure(self, query: str) -> bool:
        """Check for suspicious query structures"""
        suspicious, _ = self._has_suspicious_structure_detailed(query)
        return suspicious
    
    def _has_suspicious_structure_detailed(self, query: str) -> Tuple[bool, List[str]]:
        """Check for suspicious query structures with detailed reasons"""
        import logging
        logger = logging.getLogger(__name__)
        
        reasons = []
        
        # Count quotes - should be balanced
        single_quotes = query.count("'")
        double_quotes = query.count('"')
        
        logger.debug(f"Quote balance check: single_quotes={single_quotes}, double_quotes={double_quotes}")
        
        if single_quotes % 2 != 0:
            reasons.append(f"Unbalanced single quotes (count: {single_quotes})")
        if double_quotes % 2 != 0:
            reasons.append(f"Unbalanced double quotes (count: {double_quotes})")
        
        # Check for excessive semicolons (could indicate stacked queries)
        semicolon_count = query.count(';')
        logger.debug(f"Semicolon count: {semicolon_count}")
        if semicolon_count > 1:
            reasons.append(f"Excessive semicolons (count: {semicolon_count})")
        
        # Check for suspicious keyword combinations - but be more lenient
        suspicious_combos = [
            ('DROP', 'TABLE'), ('DELETE', 'FROM')  # Remove OR/AND/UNION which are often legitimate
        ]
        
        query_upper = query.upper()
        logger.debug(f"Checking keyword combinations in: {query_upper}")
        
        for combo in suspicious_combos:
            if all(keyword in query_upper for keyword in combo):
                logger.warning(f"Suspicious keyword combination detected: {combo}")
                reasons.append(f"Suspicious keyword combination: {' + '.join(combo)}")
        
        # Additional checks for clearly malicious patterns only
        if '1=1' in query_upper.replace(' ', ''):
            reasons.append("Always-true condition detected (1=1)")
            
        if '1<>2' in query_upper.replace(' ', '') or '1!=2' in query_upper.replace(' ', ''):
            reasons.append("Always-true condition detected (1<>2 or 1!=2)")
        
        logger.debug(f"Structure check reasons: {reasons}")
        return len(reasons) > 0, reasons
    
    def _check_correctness(self, query: str) -> ValidationResult:
        """Check for common SQL correctness issues"""
        import logging
        logger = logging.getLogger(__name__)
        
        issues = []
        corrected_query = query
        
        # Check for common error patterns
        logger.debug("Checking common error patterns...")
        for pattern, message in self.error_regex:
            if pattern.search(query):
                logger.debug(f"Found error pattern: {pattern.pattern} -> {message}")
                issues.append(message)
        
        # Basic syntax validation
        logger.debug("Performing basic syntax check...")
        syntax_ok = self._basic_syntax_check(query)
        logger.debug(f"Basic syntax check result: {syntax_ok}")
        
        if not syntax_ok:
            issues.append("Basic SQL syntax appears incorrect")
            logger.warning("Basic SQL syntax check failed")
        
        if issues:
            logger.warning(f"Correctness issues found: {issues}")
            return ValidationResult(
                status=QueryStatus.QUERY_IS_NOT_CORRECT,
                message=f"Query issues found: {'; '.join(issues)}",
                is_safe=True,
                corrected_query=corrected_query
            )
        
        logger.debug("All correctness checks passed")
        return ValidationResult(
            status=QueryStatus.QUERY_IS_CORRECT,
            message="Query validation passed",
            is_safe=True,
            corrected_query=corrected_query
        )
    
    def _basic_syntax_check(self, query: str) -> bool:
        """Basic SQL syntax validation"""
        
        # Check for balanced parentheses
        if query.count('(') != query.count(')'):
            return False
            
        # Check for basic SELECT structure - more permissive
        query_upper = query.upper().strip()
        
        # Must start with SELECT
        if not query_upper.startswith('SELECT'):
            return False
        
        # Allow three cases:
        # 1. SELECT ... FROM ... (standard queries)
        # 2. SELECT constant/function (e.g., SELECT 1, SELECT NOW())
        # 3. SELECT with subqueries
        
        # If it has FROM keyword, it should be properly structured
        if 'FROM' in query_upper:
            # Check if FROM is properly positioned (not in quotes or comments)
            if re.search(r'\bSELECT\b.*\bFROM\b', query, re.IGNORECASE | re.DOTALL):
                return True
            else:
                return False
        else:
            # SELECT without FROM is allowed for expressions and constants
            return True
    
    def get_safe_query_suggestions(self, original_query: str) -> List[str]:
        """Get suggestions for making unsafe queries safe"""
        
        suggestions = []
        
        # Convert DML to SELECT equivalents
        if re.search(r'\bINSERT\b', original_query, re.IGNORECASE):
            suggestions.append("Consider using SELECT to preview the data that would be inserted")
            
        if re.search(r'\bUPDATE\b', original_query, re.IGNORECASE):
            suggestions.append("Use SELECT to view records that would be updated")
            
        if re.search(r'\bDELETE\b', original_query, re.IGNORECASE):
            suggestions.append("Use SELECT to view records that would be deleted")
            
        if re.search(r'\b(DROP|CREATE|ALTER|TRUNCATE)\b', original_query, re.IGNORECASE):
            suggestions.append("Schema modifications are not allowed. Use SELECT to query existing tables")
        
        if re.search(r'\bUNION\s+(?:ALL\s+)?SELECT\b', original_query, re.IGNORECASE):
            suggestions.append("UNION operations might be injection attempts. Use JOINs instead if combining tables")
            
        if re.search(r'--|\*\/|\#', original_query):
            suggestions.append("Remove comments from SQL queries as they might be used for injection")
            
        if re.search(r'\bOR\s+1\s*=\s*1\b|\bAND\s+1\s*=\s*1\b', original_query, re.IGNORECASE):
            suggestions.append("Remove always-true conditions like 'OR 1=1' which are common in injection attacks")
            
        if re.search(r'\b(SLEEP|WAIT|PG_SLEEP)\s*\(', original_query, re.IGNORECASE):
            suggestions.append("Time-based functions are not allowed as they might be used for attacks")
        
        # General suggestions if no specific patterns found
        if not suggestions:
            suggestions.extend([
                "Ensure your query starts with SELECT",
                "Avoid using special characters or comments",
                "Use proper WHERE clauses to filter results",
                "Consider adding LIMIT clause to restrict result size"
            ])
        
        return suggestions
    
    def get_validation_summary(self, query: str) -> Dict[str, Any]:
        """Get comprehensive validation summary for debugging"""
        
        result = self.validate_query(query)
        
        summary = {
            "query": query,
            "is_safe": result.is_safe,
            "status": result.status.value,
            "message": result.message,
            "checks_performed": []
        }
        
        # Check which patterns were triggered
        triggered_patterns = []
        
        for pattern in self.dangerous_regex:
            if pattern.search(query):
                triggered_patterns.append(f"Dangerous pattern: {pattern.pattern}")
        
        for pattern, message in self.suspicious_regex:
            if pattern.search(query):
                triggered_patterns.append(f"Suspicious pattern: {message}")
        
        summary["triggered_patterns"] = triggered_patterns
        summary["suggestions"] = self.get_safe_query_suggestions(query) if not result.is_safe else []
        
        return summary