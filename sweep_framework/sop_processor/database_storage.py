"""
Database Storage Module

Stores extracted process flow data in a database table.
"""

import sqlite3
import json
import logging
import re
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ProcessFlowDatabase:
    """
    Manages database storage for process flow data.
    """
    
    def __init__(self, db_path: str = "process_flows.db"):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Main process flows table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS process_flows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                process_name TEXT NOT NULL,
                process_description TEXT,
                source_document TEXT NOT NULL,
                document_path TEXT,
                document_relative_path TEXT,
                extraction_model TEXT,
                extraction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                raw_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Process steps table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS process_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                process_flow_id INTEGER NOT NULL,
                step_number INTEGER NOT NULL,
                step_name TEXT NOT NULL,
                description TEXT,
                responsible_role TEXT,
                roles TEXT,
                tools TEXT,
                inputs TEXT,
                outputs TEXT,
                decision_points TEXT,
                next_steps TEXT,
                FOREIGN KEY (process_flow_id) REFERENCES process_flows(id) ON DELETE CASCADE
            )
        """)
        
        # Add roles and tools columns if they don't exist (for existing databases)
        try:
            cursor.execute("ALTER TABLE process_steps ADD COLUMN roles TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            cursor.execute("ALTER TABLE process_steps ADD COLUMN tools TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        # Roles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS process_roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                process_flow_id INTEGER NOT NULL,
                role_name TEXT NOT NULL,
                FOREIGN KEY (process_flow_id) REFERENCES process_flows(id) ON DELETE CASCADE
            )
        """)
        
        # Tools and systems table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS process_tools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                process_flow_id INTEGER NOT NULL,
                tool_name TEXT NOT NULL,
                FOREIGN KEY (process_flow_id) REFERENCES process_flows(id) ON DELETE CASCADE
            )
        """)
        
        # Compliance requirements table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compliance_requirements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                process_flow_id INTEGER NOT NULL,
                requirement TEXT NOT NULL,
                FOREIGN KEY (process_flow_id) REFERENCES process_flows(id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_process_flows_document 
            ON process_flows(source_document)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_process_steps_flow 
            ON process_steps(process_flow_id)
        """)
        
        self.conn.commit()
        logger.info("Database tables created/verified")
    
    def _parse_delimited_text(self, text: str) -> Dict[str, Any]:
        """
        Parse text-delimited process flow data into a dictionary.
        
        Args:
            text: Text-delimited string from the extractor
            
        Returns:
            Dictionary containing parsed process flow data
        """
        result = {
            'process_name': '',
            'process_description': '',
            'source_document': '',
            'document_path': '',
            'document_relative_path': '',
            'extraction_model': '',
            'extraction_temperature': None,
            'steps': [],
            'roles': set(),  # Use set to collect unique roles
            'tools_systems': set(),  # Use set to collect unique tools
            'compliance_requirements': []
        }
        
        lines = text.split('\n')
        current_section = None
        step_header_found = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse metadata headers
            if line.startswith('SOURCE_DOCUMENT:'):
                result['source_document'] = line.split(':', 1)[1].strip()
            elif line.startswith('EXTRACTION_MODEL:'):
                result['extraction_model'] = line.split(':', 1)[1].strip()
            elif line.startswith('EXTRACTION_TEMPERATURE:'):
                temp_str = line.split(':', 1)[1].strip()
                try:
                    result['extraction_temperature'] = float(temp_str)
                except ValueError:
                    pass
            elif line.startswith('DOCUMENT_PATH:'):
                result['document_path'] = line.split(':', 1)[1].strip()
            elif line.startswith('DOCUMENT_RELATIVE_PATH:'):
                result['document_relative_path'] = line.split(':', 1)[1].strip()
            
            # Parse main sections
            elif line.startswith('PROCESS_NAME:'):
                result['process_name'] = line.split(':', 1)[1].strip()
            elif line.startswith('PROCESS_DESCRIPTION:'):
                result['process_description'] = line.split(':', 1)[1].strip()
            elif line == 'STEPS:':
                current_section = 'steps'
                step_header_found = False
            elif line == 'ROLES:':
                current_section = 'roles'
            elif line == 'TOOLS_SYSTEMS:':
                current_section = 'tools'
            elif line == 'COMPLIANCE_REQUIREMENTS:':
                current_section = 'compliance'
            
            # Parse step data
            elif current_section == 'steps':
                # Check if this is the header line
                if 'STEP_NUMBER' in line and 'STEP_NAME' in line:
                    step_header_found = True
                    continue
                
                if step_header_found and '|' in line:
                    # Parse pipe-delimited step data
                    # Format: STEP_NUMBER|STEP_NAME|DESCRIPTION|RESPONSIBLE_ROLE|ROLES|TOOLS|INPUTS|OUTPUTS|DECISION_POINTS|NEXT_STEPS
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 10:
                        step = {
                            'step_number': int(parts[0]) if parts[0].isdigit() else 0,
                            'step_name': parts[1],
                            'description': parts[2],
                            'responsible_role': parts[3],
                            'roles': [r.strip() for r in parts[4].split(';') if r.strip()] if parts[4] else [],
                            'tools': [t.strip() for t in parts[5].split(';') if t.strip()] if parts[5] else [],
                            'inputs': [i.strip() for i in parts[6].split(';') if i.strip()] if parts[6] else [],
                            'outputs': [o.strip() for o in parts[7].split(';') if o.strip()] if parts[7] else [],
                            'decision_points': [d.strip() for d in parts[8].split(';') if d.strip()] if parts[8] else [],
                            'next_steps': [int(n.strip()) for n in parts[9].split(';') if n.strip().isdigit()] if parts[9] else []
                        }
                        result['steps'].append(step)
                        
                        # Collect unique roles and tools from steps
                        for role in step['roles']:
                            result['roles'].add(role)
                        if step['responsible_role']:
                            result['roles'].add(step['responsible_role'])
                        for tool in step['tools']:
                            result['tools_systems'].add(tool)
            
            # Parse roles section (if present at top level)
            elif current_section == 'roles':
                if line and not line.startswith('ROLES:'):
                    result['roles'].add(line)
            
            # Parse tools section (if present at top level)
            elif current_section == 'tools':
                if line and not line.startswith('TOOLS_SYSTEMS:'):
                    result['tools_systems'].add(line)
            
            # Parse compliance requirements
            elif current_section == 'compliance':
                if line and not line.startswith('COMPLIANCE_REQUIREMENTS:'):
                    result['compliance_requirements'].append(line)
        
        # Convert sets to lists for JSON serialization
        result['roles'] = list(result['roles'])
        result['tools_systems'] = list(result['tools_systems'])
        
        return result
    
    def insert_process_flow(self, process_flow: Union[str, Dict[str, Any]]) -> int:
        """
        Insert a process flow into the database.
        
        Args:
            process_flow: Either a text-delimited string or dictionary containing process flow data
            
        Returns:
            ID of the inserted process flow
        """
        # Parse text-delimited input if it's a string, preserving original text
        original_text = None
        if isinstance(process_flow, str):
            original_text = process_flow
            process_flow = self._parse_delimited_text(process_flow)
        
        cursor = self.conn.cursor()
        
        # Store raw data - original text if available, otherwise JSON
        raw_data = original_text if original_text else json.dumps(process_flow)
        
        # Insert main process flow record
        cursor.execute("""
            INSERT INTO process_flows (
                process_name, process_description, source_document,
                document_path, document_relative_path, extraction_model,
                raw_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            process_flow.get('process_name', ''),
            process_flow.get('process_description', ''),
            process_flow.get('source_document', ''),
            process_flow.get('document_path', ''),
            process_flow.get('document_relative_path', ''),
            process_flow.get('extraction_model', ''),
            raw_data
        ))
        
        process_flow_id = cursor.lastrowid
        
        # Insert steps
        steps = process_flow.get('steps', [])
        for step in steps:
            cursor.execute("""
                INSERT INTO process_steps (
                    process_flow_id, step_number, step_name, description,
                    responsible_role, roles, tools, inputs, outputs, decision_points, next_steps
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                process_flow_id,
                step.get('step_number', 0),
                step.get('step_name', ''),
                step.get('description', ''),
                step.get('responsible_role', ''),
                json.dumps(step.get('roles', [])),  # Store roles as JSON array
                json.dumps(step.get('tools', [])),  # Store tools as JSON array
                json.dumps(step.get('inputs', [])),
                json.dumps(step.get('outputs', [])),
                json.dumps(step.get('decision_points', [])),
                json.dumps(step.get('next_steps', []))
            ))
        
        # Insert roles (aggregated from all steps)
        roles = process_flow.get('roles', [])
        for role in roles:
            cursor.execute("""
                INSERT INTO process_roles (process_flow_id, role_name)
                VALUES (?, ?)
            """, (process_flow_id, role))
        
        # Insert tools/systems (aggregated from all steps)
        tools = process_flow.get('tools_systems', [])
        for tool in tools:
            cursor.execute("""
                INSERT INTO process_tools (process_flow_id, tool_name)
                VALUES (?, ?)
            """, (process_flow_id, tool))
        
        # Insert compliance requirements
        compliance = process_flow.get('compliance_requirements', [])
        for req in compliance:
            cursor.execute("""
                INSERT INTO compliance_requirements (process_flow_id, requirement)
                VALUES (?, ?)
            """, (process_flow_id, req))
        
        self.conn.commit()
        logger.info(f"Inserted process flow: {process_flow.get('process_name')} (ID: {process_flow_id})")
        return process_flow_id
    
    def insert_multiple(self, process_flows: List[Union[str, Dict[str, Any]]]) -> List[int]:
        """
        Insert multiple process flows into the database.
        
        Args:
            process_flows: List of process flow strings (text-delimited) or dictionaries
            
        Returns:
            List of inserted process flow IDs
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
    
    def get_process_flow(self, process_flow_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a process flow by ID.
        
        Args:
            process_flow_id: ID of the process flow
            
        Returns:
            Dictionary containing process flow data, or None if not found
        """
        cursor = self.conn.cursor()
        
        # Get main record
        cursor.execute("SELECT * FROM process_flows WHERE id = ?", (process_flow_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        # Try to parse raw_data - could be JSON or text-delimited
        raw_data = row['raw_data']
        try:
            # Try parsing as JSON first
            return json.loads(raw_data)
        except (json.JSONDecodeError, TypeError):
            # If not JSON, assume it's text-delimited and parse it
            return self._parse_delimited_text(raw_data)
    
    def list_all_processes(self) -> List[Dict[str, Any]]:
        """
        List all process flows in the database.
        
        Returns:
            List of process flow summaries
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, process_name, process_description, source_document,
                   extraction_timestamp, created_at
            FROM process_flows
            ORDER BY created_at DESC
        """)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

