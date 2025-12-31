"""
Error Logger Module

Provides error logging functionality with SQLite persistence. Captures errors,
stack traces, and optional context information for debugging and monitoring.

Example:
    >>> logger = ErrorLogger("errors.db")
    >>> try:
    ...     result = 1 / 0
    ... except Exception as e:
    ...     error_id = logger.log_error(e, context={"operation": "division"})
    >>> 
    >>> # Use as context manager
    >>> with ErrorLogger("errors.db") as logger:
    ...     logger.log_error(ValueError("Invalid input"), module="processor")
    >>> 
    >>> # Use as decorator
    >>> @logger.catch_errors
    >>> def risky_function():
    ...     return 1 / 0
"""

import sqlite3
import traceback
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorLogger:
    """
    SQLite-backed error logging system.
    
    Captures exceptions with full stack traces and optional context information,
    storing them in a local SQLite database for later analysis.
    
    Attributes:
        db_path: The resolved filesystem path to the SQLite database file.
        conn: The active SQLite connection object (set in `_connect`).
    """
    
    def __init__(self, db_path: str = "errors.db"):
        """
        Initialize the error logger and create the database table.
        
        The effective path is resolved relative to the project root unless an
        absolute `db_path` is provided.
        
        Args:
            db_path: Path to the SQLite database file. Can be relative or absolute.
                    Defaults to "errors.db" in the project root.
        
        Raises:
            sqlite3.Error: If database connection or table creation fails.
        """
        project_root = Path(__file__).resolve().parent.parent
        
        default_path = project_root / "db" / "errors.db"
        
        # Decide the effective path
        if db_path is None:
            effective = default_path
        else:
            p = Path(db_path)
            effective = p if p.is_absolute() else project_root / p
        
        # Ensure the folder exists
        effective.parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = effective
        self.conn = None
        
        self._connect()
        self._create_table()
    
    def _connect(self):
        """
        Create the SQLite connection and set row factory to `sqlite3.Row`.
        
        Sets `self.conn` and configures the row factory to enable column access
        by name.
        
        Raises:
            sqlite3.Error: If a connection cannot be established.
        """
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
    
    def _create_table(self):
        """
        Create the errors table if it doesn't exist.
        
        The table schema includes:
        - error_id: Primary key (auto-increment)
        - error_type: Type/class of the exception
        - error_message: The exception message
        - stack_trace: Full stack trace as text
        - module: Optional module name where error occurred
        - function: Optional function name where error occurred
        - context: Optional JSON context data
        - timestamp: When the error occurred (defaults to current timestamp)
        
        Raises:
            sqlite3.Error: If table creation fails.
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS errors (
                error_id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_type TEXT NOT NULL,
                error_message TEXT NOT NULL,
                stack_trace TEXT NOT NULL,
                module TEXT,
                function TEXT,
                context TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for better query performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_errors_timestamp 
            ON errors(timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_errors_type 
            ON errors(error_type)
        """)
        
        self.conn.commit()
        logger.debug(f"Error logger table created/verified at {self.db_path}")
    
    def log_error(
        self,
        error: Exception,
        module: Optional[str] = None,
        function: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Log an error to the database.
        
        Captures the exception type, message, full stack trace, and optional
        context information.
        
        Args:
            error: The exception object to log.
            module: Optional module name where the error occurred.
            function: Optional function name where the error occurred.
            context: Optional dictionary with additional context information.
        
        Returns:
            The autogenerated `error_id` of the inserted error record.
        
        Raises:
            sqlite3.Error: If insertion fails due to database issues.
        """
        import json
        
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = ''.join(traceback.format_exception(
            type(error),
            error,
            error.__traceback__
        ))
        
        # Serialize context to JSON if provided
        context_json = None
        if context:
            try:
                context_json = json.dumps(context)
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to serialize context: {e}")
                context_json = json.dumps({"serialization_error": str(e)})
        
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO errors (
                error_type,
                error_message,
                stack_trace,
                module,
                function,
                context
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            error_type,
            error_message,
            stack_trace,
            module,
            function,
            context_json
        ))
        
        error_id = cursor.lastrowid
        self.conn.commit()
        
        logger.info(
            f"Logged error: {error_type} - {error_message[:100]} "
            f"(ID: {error_id})"
        )
        
        return error_id
    
    def log_error_from_traceback(
        self,
        exc_type: type,
        exc_value: Exception,
        exc_traceback: Any,
        module: Optional[str] = None,
        function: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Log an error from sys.exc_info() or traceback information.
        
        Useful when you have exception information from sys.exc_info() or
        when handling exceptions in except blocks.
        
        Args:
            exc_type: Exception type from sys.exc_info()[0]
            exc_value: Exception value from sys.exc_info()[1]
            exc_traceback: Traceback from sys.exc_info()[2]
            module: Optional module name where the error occurred.
            function: Optional function name where the error occurred.
            context: Optional dictionary with additional context information.
        
        Returns:
            The autogenerated `error_id` of the inserted error record.
        """
        import json
        
        error_type = exc_type.__name__
        error_message = str(exc_value)
        stack_trace = ''.join(traceback.format_exception(
            exc_type,
            exc_value,
            exc_traceback
        ))
        
        # Serialize context to JSON if provided
        context_json = None
        if context:
            try:
                context_json = json.dumps(context)
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to serialize context: {e}")
                context_json = json.dumps({"serialization_error": str(e)})
        
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO errors (
                error_type,
                error_message,
                stack_trace,
                module,
                function,
                context
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            error_type,
            error_message,
            stack_trace,
            module,
            function,
            context_json
        ))
        
        error_id = cursor.lastrowid
        self.conn.commit()
        
        logger.info(
            f"Logged error: {error_type} - {error_message[:100]} "
            f"(ID: {error_id})"
        )
        
        return error_id
    
    def get_error(self, error_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve an error record by ID.
        
        Args:
            error_id: The ID of the error record to retrieve.
        
        Returns:
            A dictionary containing the error record, or None if not found.
            Keys: error_id, error_type, error_message, stack_trace, module,
                  function, context (as JSON string), timestamp.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM errors WHERE error_id = ?", (error_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def list_errors(
        self,
        limit: Optional[int] = None,
        error_type: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> list[Dict[str, Any]]:
        """
        List error records from the database.
        
        Args:
            limit: Maximum number of records to return (None for all).
            error_type: Filter by error type/class name (case-sensitive).
            since: Filter errors after this datetime.
        
        Returns:
            A list of dictionaries, each representing an error record.
        """
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM errors WHERE 1=1"
        params = []
        
        if error_type:
            query += " AND error_type = ?"
            params.append(error_type)
        
        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def count_errors(
        self,
        error_type: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> int:
        """
        Count error records matching the criteria.
        
        Args:
            error_type: Filter by error type/class name (case-sensitive).
            since: Filter errors after this datetime.
        
        Returns:
            The number of matching error records.
        """
        cursor = self.conn.cursor()
        
        query = "SELECT COUNT(*) as count FROM errors WHERE 1=1"
        params = []
        
        if error_type:
            query += " AND error_type = ?"
            params.append(error_type)
        
        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())
        
        cursor.execute(query, params)
        result = cursor.fetchone()
        return result['count'] if result else 0
    
    def catch_errors(
        self,
        module: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Decorator to automatically log exceptions from a function.
        
        Usage:
            >>> logger = ErrorLogger()
            >>> @logger.catch_errors(module="my_module")
            >>> def my_function():
            ...     raise ValueError("Something went wrong")
        
        Args:
            module: Optional module name to associate with logged errors.
            context: Optional context dictionary to include with all errors
                    from this function. Can be overridden by context passed
                    in the decorated function.
        
        Returns:
            A decorator function that wraps the target function.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Merge decorator context with any context from kwargs
                    error_context = context.copy() if context else {}
                    if 'error_context' in kwargs:
                        error_context.update(kwargs.pop('error_context'))
                    
                    self.log_error(
                        e,
                        module=module or func.__module__,
                        function=func.__name__,
                        context=error_context if error_context else None
                    )
                    raise  # Re-raise the exception after logging
            return wrapper
        return decorator
    
    def close(self):
        """
        Close the database connection if open.
        
        Ensures that the SQLite connection is closed and logs the event.
        """
        if self.conn:
            self.conn.close()
            logger.debug("Error logger database connection closed")
    
    def __enter__(self):
        """
        Enter the runtime context and return the error logger instance.
        
        Returns:
            The current `ErrorLogger` instance with an active connection.
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context, optionally log any exception, and close the connection.
        
        If an exception occurred within the context, it will be logged automatically
        before closing the connection.
        
        Args:
            exc_type: Exception type if raised within the context.
            exc_val: Exception value if raised within the context.
            exc_tb: Traceback if raised within the context.
        
        Side Effects:
            Logs any exception that occurred, then always calls `close()`.
        """
        if exc_type is not None:
            self.log_error_from_traceback(
                exc_type,
                exc_val,
                exc_tb
            )
        self.close()

