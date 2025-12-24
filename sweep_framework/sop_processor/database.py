
"""
Database storage module.

Stores extracted process flow data in a SQLite database. This module provides
a lightweight persistence layer with helpers to insert single or multiple
process flows and list previously stored processes.

Schema Assumptions:
    The following tables are expected to exist:

    - process(process_id INTEGER PRIMARY KEY AUTOINCREMENT,
              process_name TEXT,
              process_description TEXT,
              source_document TEXT,
              document_path TEXT,
              extraction_model TEXT,
              extraction_timestamp TEXT,
              created_at TEXT)

    - step(step_id INTEGER PRIMARY KEY AUTOINCREMENT,
           process_id INTEGER REFERENCES process(process_id),
           step_number INTEGER,
           step_name TEXT,
           step_description TEXT,
           responsible_role TEXT,
           inputs TEXT,          -- comma-separated values
           outputs TEXT,         -- comma-separated values
           tools TEXT,           -- comma-separated values
           decision_points TEXT, -- comma-separated values
           next_steps TEXT)      -- comma-separated step numbers (string)

Example:
    >>> db = Database("db/stream.db")
    >>> process_id = db.insert_process_flow({
    ...     "process_name": "Year-End Processing",
    ...     "process_description": "Batch process for annual statements.",
    ...     "source_document": "yearend.pdf",
    ...     "document_path": "/docs/yearend.pdf",
    ...     "extraction_model": "gpt-4o",
    ...     "steps": [
    ...         {
    ...             "step_number": 1,
    ...             "step_name": "Collect Inputs",
    ...             "description": "Gather all client data.",
    ...             "responsible_role": "Operations Analyst",
    ...             "inputs": ["client_master", "previous_statements"],
    ...             "outputs": ["input_package"],
    ...             "tools": ["Excel", "ETL"],
    ...             "decision_points": ["data_complete?"],
    ...             "next_steps": [2]
    ...         }
    ...     ]
    ... })
    >>> rows = db.list_all_processes()
"""

import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class Database:
    """
    SQLite-backed storage for process flow data.

    Resolves the effective database path relative to the project root by default,
    ensures the containing folder exists, and manages a connection lifecycle.

    Attributes:
        db_path: The resolved filesystem path to the SQLite database file.
        conn: The active SQLite connection object (set in `_connect`).
    """

    def __init__(self, db_path: str = "stream.db"):
        """
        Initialize the database client and open a connection.

        The effective path is resolved relative to the project root unless an
        absolute `db_path` is provided. If `db_path` is `None`, a default path
        of `<project_root>/db/stream.db` is used.

        Args:
            db_path: Path to the SQLite database file. Can be relative or absolute.

        Raises:
            Exception: Propagates any underlying filesystem or sqlite errors when
                creating directories or connecting to the database.
        """

        project_root = Path(__file__).resolve().parent.parent

        default_path = project_root / "db" / "stream.db"

        # Decide the effective path

        if db_path is None:
            effective = default_path
        else:
            p = Path(db_path)
            effective = p if p.is_absolute() else project_root / p

        # Ensure the folder exists and the file is locally available

        effective.parent.mkdir(parents = True, exist_ok = True)

        self.db_path = effective

        self.conn = None

        self._connect()
    
    def _connect(self):
        """
        Create the SQLite connection and set row factory to `sqlite3.Row`.

        Sets `self.conn` and configures the row factory to enable column access
        by name.

        Raises:
            sqlite3.Error: If a connection cannot be established.
        """

        self.conn = sqlite3.connect(self.db_path)

        self.conn.row_factory = sqlite3.Row  # Enable column access by name
    
    def insert_process_flow(self, process_flow: Dict[str, Any]) -> int:
        """
        Insert a single process flow and its steps.

        Expects a dictionary with at least the following keys:
            - process_name (str)
            - process_description (str)
            - source_document (str)
            - document_path (str)
            - extraction_model (str)
            - steps (list[dict]) with each step having:
                * step_number (int)
                * step_name (str)
                * description (str)
                * responsible_role (str)
                * inputs (list[str])
                * outputs (list[str])
                * tools (list[str])
                * decision_points (list[str])
                * next_steps (list[int])

        Args:
            process_flow: A dictionary representing the process flow. See the
                structure above for expected keys and value types.

        Returns:
            The autogenerated `process_id` of the inserted process.

        Raises:
            sqlite3.Error: If insertion fails due to schema or constraint issues.
            KeyError: If required keys are missing from `process_flow`.
            ValueError: If types are incorrect (e.g., steps not a list).
        """

        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO process (
                process_name, 
                process_description, 
                source_document,
                document_path, 
                extraction_model
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            process_flow.get('process_name', ''),
            process_flow.get('process_description', ''),
            process_flow.get('source_document', ''),
            process_flow.get('document_path', ''),
            process_flow.get('extraction_model', ''),
        ))
        
        process_id = cursor.lastrowid
        
        steps = process_flow.get('steps', [])

        for step in steps:
            cursor.execute("""
                INSERT INTO step (
                    process_id, 
                    step_number, 
                    step_name, 
                    step_description,
                    responsible_role, 
                    inputs, 
                    outputs, 
                    tools,
                    decision_points, 
                    next_steps
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                process_id,
                step.get('step_number', 0),
                step.get('step_name', ''),
                step.get('description', ''),
                step.get('responsible_role', ''),
                ", ".join(step.get('inputs', [])),
                ", ".join(step.get('outputs', [])),
                ", ".join(step.get('tools', [])),
                ", ".join(step.get('decision_points', [])),
                ", ".join(map(str, step.get('next_steps', [])))
            ))
        
        self.conn.commit()

        logger.info(f"Inserted process flow: {process_flow.get('process_name')} (ID: {process_id})")

        return process_id
    
    def insert_multiple(self, process_flows: List[Dict[str, Any]]) -> List[int]:
        """
        Insert multiple process flows, continuing on individual failures.

        Args:
            process_flows: A list of process flow dictionaries. Each item must
                match the expected shape documented in `insert_process_flow`.

        Returns:
            A list of successfully inserted process IDs.

        Notes:
            Any failures are logged. The method continues inserting subsequent
            flows even if one of them fails.
        """

        ids = []

        for flow in process_flows:
            try:
                flow_id = self.insert_process_flow(flow)
                ids.append(flow_id)
            except Exception as e:
                logger.error(f"Failed to insert process flow: {e}")
                continue
        return ids
    
    def list_all_processes(self) -> List[Dict[str, Any]]:
        """
        List all processes ordered by `created_at` descending.

        Returns:
            A list of dictionaries, each representing a row from the `process` table with
            keys: `process_id`, `process_name`, `process_description`, `source_document`,
            `extraction_timestamp`, and `created_at`.

        Raises:
            sqlite3.Error: If the query fails due to missing table or columns.
        """

        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT process_id, process_name, process_description, source_document,
                   extraction_timestamp, created_at
            FROM process
            ORDER BY created_at DESC
        """)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """
        Close the database connection if open.

        Ensures that the SQLite connection is closed and logs the event.
        """

        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """
        Enter the runtime context and return the database instance.

        Returns:
            The current `Database` instance with an active connection.
        """

        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context and close the connection.

        Args:
            exc_type: Exception type if raised within the context.
            exc_val: Exception value if raised within the context.
            exc_tb: Traceback if raised within the context.

        Side Effects:
            Always calls `close()` to ensure the connection is released.
        """

        self.close()
