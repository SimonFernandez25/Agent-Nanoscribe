# ======================================================
# File: gwl_validator.py
# GWL Validator implementing langium grammar rules
# ======================================================

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class ValidationError:
    """Represents a single validation error."""
    line: int
    message: str
    severity: str = "error"  # "error" or "warning"
    
    def to_dict(self) -> Dict:
        return {
            "line": self.line,
            "message": self.message,
            "severity": self.severity
        }


@dataclass
class FileValidation:
    """Validation result for a single file."""
    file: str
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "file": self.file,
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors]
        }


@dataclass
class ValidationReport:
    """Complete validation report for all files."""
    valid: bool
    validations: List[FileValidation] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "valid": self.valid,
            "validations": [v.to_dict() for v in self.validations]
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class GWLValidator:
    """
    Validates GWL code against the langium grammar specification.
    
    Grammar rules implemented:
    - VarDecl: var|local $name = expr
    - Assignment: set $name = expr
    - ForLoop: for $var = expr to expr [step expr] ... end
    - IfBlock: if condition ... [else ...] end
    - IncludeStmt: include filepath
    - CoordinateLine: expr expr expr [expr]
    - CmdStmt: command [args]
    """
    
    # Terminal patterns from langium grammar
    NUMBER = r'[0-9]+(\.[0-9]+)?'
    STRING = r'"[^"]*"'
    ID = r'[_a-zA-Z][_a-zA-Z0-9]*'
    VAR_NAME = r'\$[_a-zA-Z][_a-zA-Z0-9]*'
    COMMENT = r'%[^\n\r]*'
    
    # Keywords
    KEYWORDS = {'var', 'local', 'set', 'for', 'to', 'step', 'end', 'if', 'else', 'include'}
    
    # Built-in functions (common ones)
    FUNCTIONS = {'sin', 'cos', 'tan', 'sqrt', 'abs', 'ceil', 'floor', 'mod', 'pow', 'exp', 'log'}
    
    # Known commands (common Nanoscribe/DescribeX commands)
    COMMANDS = {
        'XOffset', 'YOffset', 'ZOffset', 'AddZOffset',
        'PowerScaling', 'LaserPower', 'ScanSpeed', 'LineNumber',
        'MoveStageX', 'MoveStageY', 'MoveStageZ',
        'FindInterfaceAt', 'Write', 'Include',
        'PiezoSettlingTime', 'GalvoSettlingTime'
    }
    
    def __init__(self):
        self.errors: List[ValidationError] = []
        self.lines: List[str] = []
        self.block_stack: List[Tuple[str, int]] = []  # (type, line_number)
        self.declared_vars: set = set()
    
    def validate_file(self, filepath: str) -> FileValidation:
        """Validate a single GWL file."""
        path = Path(filepath)
        if not path.exists():
            return FileValidation(
                file=path.name,
                valid=False,
                errors=[ValidationError(0, f"File not found: {filepath}")]
            )
        
        content = path.read_text(encoding='utf-8')
        return self.validate_content(content, path.name)
    
    def validate_content(self, content: str, filename: str = "unknown.gwl") -> FileValidation:
        """Validate GWL content string."""
        self.errors = []
        self.lines = content.split('\n')
        self.block_stack = []
        self.declared_vars = set()
        
        for line_num, line in enumerate(self.lines, 1):
            self._validate_line(line, line_num)
        
        # Check for unclosed blocks
        for block_type, block_line in self.block_stack:
            self.errors.append(ValidationError(
                block_line,
                f"Unclosed '{block_type}' block starting at line {block_line}"
            ))
        
        is_valid = len(self.errors) == 0
        return FileValidation(file=filename, valid=is_valid, errors=self.errors)
    
    def _validate_line(self, line: str, line_num: int):
        """Validate a single line of GWL code."""
        # Strip whitespace and handle empty lines
        stripped = line.strip()
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith('%'):
            return
        
        # Remove inline comments
        if '%' in stripped:
            stripped = stripped[:stripped.index('%')].strip()
            if not stripped:
                return
        
        # Try to match each statement type
        if self._is_var_decl(stripped, line_num):
            return
        if self._is_assignment(stripped, line_num):
            return
        if self._is_for_loop(stripped, line_num):
            return
        if self._is_if_block(stripped, line_num):
            return
        if self._is_else(stripped, line_num):
            return
        if self._is_end(stripped, line_num):
            return
        if self._is_include(stripped, line_num):
            return
        if self._is_coordinate_line(stripped, line_num):
            return
        if self._is_command(stripped, line_num):
            return
        
        # If nothing matched, it might be an error
        self.errors.append(ValidationError(
            line_num,
            f"Unrecognized statement: '{stripped[:50]}...' " if len(stripped) > 50 else f"Unrecognized statement: '{stripped}'",
            severity="warning"
        ))
    
    def _is_var_decl(self, line: str, line_num: int) -> bool:
        """Check for: var|local $name = expr"""
        pattern = rf'^(var|local)\s+(\${self.ID})\s*=\s*(.+)$'
        match = re.match(pattern, line)
        if match:
            var_name = match.group(2)
            expr = match.group(3)
            self.declared_vars.add(var_name)
            self._validate_expression(expr, line_num)
            return True
        return False
    
    def _is_assignment(self, line: str, line_num: int) -> bool:
        """Check for: set $name = expr"""
        pattern = rf'^set\s+(\${self.ID})\s*=\s*(.+)$'
        match = re.match(pattern, line)
        if match:
            var_name = match.group(1)
            expr = match.group(2)
            self._validate_expression(expr, line_num)
            return True
        return False
    
    def _is_for_loop(self, line: str, line_num: int) -> bool:
        """Check for: for $var = expr to expr [step expr]"""
        pattern = rf'^for\s+(\${self.ID})\s*=\s*(.+?)\s+to\s+(.+?)(?:\s+step\s+(.+))?$'
        match = re.match(pattern, line)
        if match:
            var_name = match.group(1)
            from_expr = match.group(2)
            to_expr = match.group(3)
            step_expr = match.group(4)
            
            self.block_stack.append(('for', line_num))
            self._validate_expression(from_expr, line_num)
            self._validate_expression(to_expr, line_num)
            if step_expr:
                self._validate_expression(step_expr, line_num)
            return True
        return False
    
    def _is_if_block(self, line: str, line_num: int) -> bool:
        """Check for: if condition"""
        pattern = rf'^if\s+(.+)$'
        match = re.match(pattern, line)
        if match:
            condition = match.group(1)
            self.block_stack.append(('if', line_num))
            self._validate_condition(condition, line_num)
            return True
        return False
    
    def _is_else(self, line: str, line_num: int) -> bool:
        """Check for: else"""
        if line == 'else':
            if not self.block_stack or self.block_stack[-1][0] != 'if':
                self.errors.append(ValidationError(
                    line_num,
                    "'else' without matching 'if'"
                ))
            return True
        return False
    
    def _is_end(self, line: str, line_num: int) -> bool:
        """Check for: end"""
        if line == 'end':
            if not self.block_stack:
                self.errors.append(ValidationError(
                    line_num,
                    "'end' without matching 'for' or 'if'"
                ))
            else:
                self.block_stack.pop()
            return True
        return False
    
    def _is_include(self, line: str, line_num: int) -> bool:
        """Check for: include filepath"""
        pattern = rf'^[Ii]nclude\s+(.+)$'
        match = re.match(pattern, line)
        if match:
            filepath = match.group(1).strip()
            # Validate filepath format
            if not re.match(rf'^[\w./\\-]+\.gwl$', filepath):
                self.errors.append(ValidationError(
                    line_num,
                    f"Invalid include path: '{filepath}'",
                    severity="warning"
                ))
            return True
        return False
    
    def _is_coordinate_line(self, line: str, line_num: int) -> bool:
        """Check for: expr expr expr [expr] (coordinate line)"""
        # A coordinate line starts with a number or variable, has 3-4 expressions
        parts = self._split_expressions(line)
        if len(parts) >= 3 and len(parts) <= 4:
            # First part should not be a bare ID (command)
            first = parts[0].strip()
            if first.startswith('$') or re.match(rf'^-?{self.NUMBER}', first) or first.startswith('('):
                for part in parts:
                    self._validate_expression(part, line_num)
                return True
        return False
    
    def _is_command(self, line: str, line_num: int) -> bool:
        """Check for: CommandName [args]"""
        parts = line.split(None, 1)
        if not parts:
            return False
        
        cmd = parts[0]
        
        # Check if it's a known command or looks like a command (starts with uppercase or is ID)
        if cmd in self.COMMANDS or (re.match(rf'^{self.ID}$', cmd) and cmd not in self.KEYWORDS):
            # Validate arguments if present
            if len(parts) > 1:
                args = parts[1]
                # Handle format args: # (arg1, arg2)
                if '#' in args:
                    main_args, format_part = args.split('#', 1)
                    for arg in main_args.split():
                        self._validate_expression(arg, line_num)
                else:
                    for arg in args.split():
                        self._validate_expression(arg, line_num)
            return True
        return False
    
    def _split_expressions(self, line: str) -> List[str]:
        """Split a line into expression tokens."""
        # Simple split by whitespace, handling parentheses
        tokens = []
        current = ""
        paren_depth = 0
        
        for char in line:
            if char == '(':
                paren_depth += 1
                current += char
            elif char == ')':
                paren_depth -= 1
                current += char
            elif char.isspace() and paren_depth == 0:
                if current:
                    tokens.append(current)
                    current = ""
            else:
                current += char
        
        if current:
            tokens.append(current)
        
        return tokens
    
    def _validate_expression(self, expr: str, line_num: int):
        """Validate an expression."""
        expr = expr.strip()
        if not expr:
            return
        
        # Check for balanced parentheses
        if expr.count('(') != expr.count(')'):
            self.errors.append(ValidationError(
                line_num,
                f"Unbalanced parentheses in expression: '{expr}'"
            ))
        
        # Check for valid tokens
        # Remove parentheses and operators for token validation
        tokens = re.findall(rf'{self.VAR_NAME}|{self.NUMBER}|{self.ID}', expr)
        
        for token in tokens:
            if token.startswith('$'):
                # Variable reference - OK
                pass
            elif re.match(rf'^{self.NUMBER}$', token):
                # Number - OK
                pass
            elif token in self.FUNCTIONS or token in self.KEYWORDS:
                # Known function or keyword - OK
                pass
            elif re.match(rf'^{self.ID}$', token):
                # Unknown identifier - could be a function call
                pass
    
    def _validate_condition(self, condition: str, line_num: int):
        """Validate a condition (left op right)."""
        # Check for comparison operator
        ops = ['==', '!=', '<=', '>=', '<', '>']
        found_op = False
        for op in ops:
            if op in condition:
                found_op = True
                parts = condition.split(op, 1)
                if len(parts) == 2:
                    self._validate_expression(parts[0], line_num)
                    self._validate_expression(parts[1], line_num)
                break
        
        if not found_op:
            self.errors.append(ValidationError(
                line_num,
                f"Condition missing comparison operator: '{condition}'"
            ))


def validate_gwl_files(filepaths: List[str]) -> ValidationReport:
    """
    Validate multiple GWL files and return a combined report.
    
    Args:
        filepaths: List of paths to GWL files
        
    Returns:
        ValidationReport with results for all files
    """
    validator = GWLValidator()
    validations = []
    all_valid = True
    
    for filepath in filepaths:
        result = validator.validate_file(filepath)
        validations.append(result)
        if not result.valid:
            all_valid = False
    
    return ValidationReport(valid=all_valid, validations=validations)


def validate_gwl_content(contents: Dict[str, str]) -> ValidationReport:
    """
    Validate GWL content from strings.
    
    Args:
        contents: Dict mapping filename to GWL content string
        
    Returns:
        ValidationReport with results for all content
    """
    validator = GWLValidator()
    validations = []
    all_valid = True
    
    for filename, content in contents.items():
        result = validator.validate_content(content, filename)
        validations.append(result)
        if not result.valid:
            all_valid = False
    
    return ValidationReport(valid=all_valid, validations=validations)


# ======================================================
# CLI Usage
# ======================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python gwl_validator.py <file1.gwl> [file2.gwl ...]")
        sys.exit(1)
    
    files = sys.argv[1:]
    report = validate_gwl_files(files)
    
    print(report.to_json())
    
    if not report.valid:
        sys.exit(1)
