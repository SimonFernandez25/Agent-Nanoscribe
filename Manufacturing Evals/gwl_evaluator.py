# ======================================================
# File: gwl_evaluator.py
# Headless GWL Print Time Estimator
# ======================================================

import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json


@dataclass
class EvaluationMetrics:
    """Metrics extracted from GWL evaluation."""
    total_write_length_um: float = 0.0
    write_segment_count: int = 0
    z_layer_count: int = 0
    z_offset_total_um: float = 0.0
    stage_move_count: int = 0
    stage_move_distance_um: float = 0.0
    estimated_time_sec: float = 0.0
    
    # Breakdown
    write_time_sec: float = 0.0
    z_offset_time_sec: float = 0.0
    stage_move_time_sec: float = 0.0
    overhead_time_sec: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "total_write_length_um": round(self.total_write_length_um, 2),
            "write_segment_count": self.write_segment_count,
            "z_layer_count": self.z_layer_count,
            "z_offset_total_um": round(self.z_offset_total_um, 2),
            "stage_move_count": self.stage_move_count,
            "stage_move_distance_um": round(self.stage_move_distance_um, 2),
            "estimated_time_sec": round(self.estimated_time_sec, 2),
            "time_breakdown": {
                "write_sec": round(self.write_time_sec, 2),
                "z_offset_sec": round(self.z_offset_time_sec, 2),
                "stage_move_sec": round(self.stage_move_time_sec, 2),
                "overhead_sec": round(self.overhead_time_sec, 2)
            }
        }


class GWLEvaluator:
    """
    Headless GWL evaluator for print time estimation.
    
    Calibration targets:
    - meta_1.gwl: ~17 seconds
    - main_array.gwl (5x5): ~458 seconds
    """
    
    # Timing constants (calibrated to real measurements)
    # meta_1.gwl ~17s, main_array.gwl (5x5) ~458s
    SCAN_SPEED_UM_PER_SEC = 100.0  # Default ScanSpeed (from GWL files)
    Z_OFFSET_TIME_SEC = 0.15  # Time per AddZOffset command
    STAGE_MOVE_SPEED_UM_PER_SEC = 500.0  # Stage movement speed
    STAGE_SETTLE_TIME_SEC = 0.35  # Settling time after stage move
    OVERHEAD_PER_META_SEC = 0.3  # Overhead per meta Include
    BASE_OVERHEAD_SEC = 2.0  # Base overhead for any print
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path.cwd()
        self.variables: Dict[str, float] = {}
        self.current_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.current_z_offset: float = 0.0
        self.scan_speed: float = self.SCAN_SPEED_UM_PER_SEC
        self.pending_coords: List[Tuple[float, float, float]] = []
        
        # Metrics
        self.metrics = EvaluationMetrics()
        
        # Built-in functions
        self.functions = {
            'sqrt': math.sqrt,
            'sin': lambda x: math.sin(math.radians(x)),
            'cos': lambda x: math.cos(math.radians(x)),
            'tan': lambda x: math.tan(math.radians(x)),
            'abs': abs,
            'ceil': math.ceil,
            'floor': math.floor,
            'mod': lambda a, b: a % b,
            'pow': math.pow,
            'exp': math.exp,
            'log': math.log,
        }
    
    def evaluate_file(self, filepath: str) -> EvaluationMetrics:
        """Evaluate a GWL file and return metrics."""
        path = Path(filepath)
        if not path.is_absolute():
            path = self.base_dir / path
        
        if not path.exists():
            raise FileNotFoundError(f"GWL file not found: {path}")
        
        self.base_dir = path.parent
        content = path.read_text(encoding='utf-8')
        
        # Reset state
        self._reset()
        
        # Parse and execute
        self._execute(content)
        
        # Calculate timing
        self._calculate_timing()
        
        return self.metrics
    
    def _reset(self):
        """Reset evaluator state."""
        self.variables = {}
        self.current_position = (0.0, 0.0, 0.0)
        self.current_z_offset = 0.0
        self.scan_speed = self.SCAN_SPEED_UM_PER_SEC
        self.pending_coords = []
        self.metrics = EvaluationMetrics()
    
    def _execute(self, content: str):
        """Execute GWL content."""
        lines = content.split('\n')
        self._execute_lines(lines)
    
    def _execute_lines(self, lines: List[str]):
        """Execute a list of GWL lines."""
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('%'):
                i += 1
                continue
            
            # Remove inline comments
            if '%' in line:
                line = line[:line.index('%')].strip()
                if not line:
                    i += 1
                    continue
            
            # Handle different statement types
            if line.startswith('var ') or line.startswith('local '):
                self._handle_var_decl(line)
            elif line.startswith('set '):
                self._handle_assignment(line)
            elif line.startswith('for '):
                # Find matching end
                end_idx = self._find_matching_end(lines, i)
                body = lines[i+1:end_idx]
                self._handle_for_loop(line, body)
                i = end_idx
            elif line.startswith('if '):
                end_idx = self._find_matching_end(lines, i)
                body = lines[i+1:end_idx]
                self._handle_if_block(line, body)
                i = end_idx
            elif line.startswith('Include ') or line.startswith('include '):
                self._handle_include(line)
            elif line == 'Write':
                self._handle_write()
            elif line == 'end':
                pass  # Handled by for/if
            elif self._is_coordinate_line(line):
                self._handle_coordinate(line)
            else:
                self._handle_command(line)
            
            i += 1
    
    def _find_matching_end(self, lines: List[str], start: int) -> int:
        """Find matching 'end' for a block starting at start index."""
        depth = 1
        i = start + 1
        while i < len(lines) and depth > 0:
            line = lines[i].strip()
            if line.startswith('for ') or line.startswith('if '):
                depth += 1
            elif line == 'end':
                depth -= 1
            i += 1
        return i - 1
    
    def _handle_var_decl(self, line: str):
        """Handle: var|local $name = expr"""
        match = re.match(r'(?:var|local)\s+\$(\w+)\s*=\s*(.+)', line)
        if match:
            name = match.group(1)
            expr = match.group(2)
            self.variables[name] = self._eval_expr(expr)
    
    def _handle_assignment(self, line: str):
        """Handle: set $name = expr"""
        match = re.match(r'set\s+\$(\w+)\s*=\s*(.+)', line)
        if match:
            name = match.group(1)
            expr = match.group(2)
            self.variables[name] = self._eval_expr(expr)
    
    def _handle_for_loop(self, header: str, body: List[str]):
        """Handle: for $var = start to end [step val]"""
        match = re.match(
            r'for\s+\$(\w+)\s*=\s*(.+?)\s+to\s+(.+?)(?:\s+step\s+(.+))?$',
            header
        )
        if not match:
            return
        
        var_name = match.group(1)
        start_val = self._eval_expr(match.group(2))
        end_val = self._eval_expr(match.group(3))
        step_val = self._eval_expr(match.group(4)) if match.group(4) else 1.0
        
        if step_val == 0:
            return
        
        # Execute loop
        current = start_val
        if step_val > 0:
            while current <= end_val + 1e-9:
                self.variables[var_name] = current
                self._execute_lines(body)
                current += step_val
        else:
            while current >= end_val - 1e-9:
                self.variables[var_name] = current
                self._execute_lines(body)
                current += step_val
    
    def _handle_if_block(self, header: str, body: List[str]):
        """Handle: if condition ... [else ...] end"""
        match = re.match(r'if\s+(.+)', header)
        if not match:
            return
        
        condition = match.group(1)
        
        # Find else
        else_idx = None
        depth = 0
        for i, line in enumerate(body):
            stripped = line.strip()
            if stripped.startswith('for ') or stripped.startswith('if '):
                depth += 1
            elif stripped == 'end':
                depth -= 1
            elif stripped == 'else' and depth == 0:
                else_idx = i
                break
        
        if self._eval_condition(condition):
            if else_idx is not None:
                self._execute_lines(body[:else_idx])
            else:
                self._execute_lines(body)
        else:
            if else_idx is not None:
                self._execute_lines(body[else_idx+1:])
    
    def _handle_include(self, line: str):
        """Handle: Include filepath"""
        match = re.match(r'[Ii]nclude\s+(.+)', line)
        if match:
            filepath = match.group(1).strip()
            full_path = self.base_dir / filepath
            if full_path.exists():
                content = full_path.read_text(encoding='utf-8')
                
                # Save parent scope state
                parent_vars = self.variables.copy()
                parent_z = self.current_z_offset
                parent_pending = self.pending_coords.copy()
                
                # Reset Z offset for included file (starts fresh)
                self.current_z_offset = 0.0
                self.pending_coords = []
                
                # Execute included file (metrics accumulate automatically)
                self._execute(content)
                
                # Restore parent scope
                self.variables = parent_vars
                self.current_z_offset = parent_z
                self.pending_coords = parent_pending
                
                # Count as one meta overhead
                self.metrics.overhead_time_sec += self.OVERHEAD_PER_META_SEC
    
    def _handle_command(self, line: str):
        """Handle GWL commands."""
        parts = line.split()
        if not parts:
            return
        
        cmd = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd == 'ScanSpeed' and args:
            self.scan_speed = self._eval_expr(args[0])
        elif cmd == 'AddZOffset' and args:
            offset = self._eval_expr(args[0])
            self.current_z_offset += offset
            self.metrics.z_layer_count += 1
            self.metrics.z_offset_total_um += abs(offset)
        elif cmd == 'ZOffset' and args:
            self.current_z_offset = self._eval_expr(args[0])
        elif cmd == 'MoveStageX' and args:
            dx = abs(self._eval_expr(args[0]))
            self.metrics.stage_move_count += 1
            self.metrics.stage_move_distance_um += dx
        elif cmd == 'MoveStageY' and args:
            dy = abs(self._eval_expr(args[0]))
            self.metrics.stage_move_count += 1
            self.metrics.stage_move_distance_um += dy
        elif cmd in ('XOffset', 'YOffset', 'PowerScaling', 'LaserPower', 
                     'FindInterfaceAt', 'LineNumber'):
            pass  # Ignore for timing purposes
    
    def _is_coordinate_line(self, line: str) -> bool:
        """Check if line is a coordinate triplet."""
        parts = line.split()
        if len(parts) < 3 or len(parts) > 4:
            return False
        
        # First part should start with number, $, or (
        first = parts[0]
        if first[0].isalpha() and not first.startswith('$'):
            return False
        
        return True
    
    def _handle_coordinate(self, line: str):
        """Handle coordinate triplet."""
        parts = line.split()
        if len(parts) >= 3:
            x = self._eval_expr(parts[0])
            y = self._eval_expr(parts[1])
            z = self._eval_expr(parts[2]) + self.current_z_offset
            self.pending_coords.append((x, y, z))
    
    def _handle_write(self):
        """Handle Write command - calculate toolpath length."""
        if len(self.pending_coords) >= 2:
            # Calculate polyline length
            for i in range(1, len(self.pending_coords)):
                p1 = self.pending_coords[i-1]
                p2 = self.pending_coords[i]
                dist = math.sqrt(
                    (p2[0] - p1[0])**2 + 
                    (p2[1] - p1[1])**2 + 
                    (p2[2] - p1[2])**2
                )
                self.metrics.total_write_length_um += dist
            
            self.metrics.write_segment_count += 1
            self.current_position = self.pending_coords[-1]
        
        self.pending_coords = []
    
    def _eval_expr(self, expr: str) -> float:
        """Evaluate a GWL expression."""
        if not expr:
            return 0.0
        
        expr = expr.strip()
        
        # Handle simple number
        try:
            return float(expr)
        except ValueError:
            pass
        
        # Handle simple variable reference
        if expr.startswith('$') and re.match(r'^\$\w+$', expr):
            var_name = expr[1:]
            return self.variables.get(var_name, 0.0)
        
        # Handle negated variable
        if expr.startswith('-$') and re.match(r'^-\$\w+$', expr):
            var_name = expr[2:]
            return -self.variables.get(var_name, 0.0)
        
        # Substitute all variables in the expression
        def replace_var(match):
            var_name = match.group(1)
            val = self.variables.get(var_name, 0.0)
            # Wrap in parentheses to preserve order of operations
            return f'({val})'
        
        expr = re.sub(r'\$(\w+)', replace_var, expr)
        
        # Handle function calls with balanced parentheses
        max_iterations = 20
        for _ in range(max_iterations):
            found_func = False
            for func_name, func in self.functions.items():
                # Find function call start
                pattern = rf'\b{func_name}\s*\('
                match = re.search(pattern, expr)
                if match:
                    # Find matching closing paren
                    start = match.end() - 1  # Position of opening paren
                    depth = 1
                    end = start + 1
                    while end < len(expr) and depth > 0:
                        if expr[end] == '(':
                            depth += 1
                        elif expr[end] == ')':
                            depth -= 1
                        end += 1
                    
                    if depth == 0:
                        found_func = True
                        args_str = expr[start+1:end-1]  # Content between parens
                        
                        # Evaluate the inner expression first, then apply function
                        try:
                            inner_val = self._safe_eval(args_str)
                            result = func(inner_val)
                            expr = expr[:match.start()] + str(result) + expr[end:]
                        except Exception as e:
                            expr = expr[:match.start()] + '0' + expr[end:]
                        break
            if not found_func:
                break
        
        return self._safe_eval(expr)
    
    def _safe_eval(self, expr: str) -> float:
        """Safely evaluate a mathematical expression."""
        try:
            # Only allow safe operations
            result = eval(expr, {"__builtins__": {}}, {})
            return float(result)
        except Exception:
            return 0.0
    
    def _eval_condition(self, condition: str) -> bool:
        """Evaluate a condition."""
        ops = ['==', '!=', '<=', '>=', '<', '>']
        for op in ops:
            if op in condition:
                parts = condition.split(op, 1)
                if len(parts) == 2:
                    left = self._eval_expr(parts[0])
                    right = self._eval_expr(parts[1])
                    if op == '==':
                        return left == right
                    elif op == '!=':
                        return left != right
                    elif op == '<=':
                        return left <= right
                    elif op == '>=':
                        return left >= right
                    elif op == '<':
                        return left < right
                    elif op == '>':
                        return left > right
        return False
    
    def _calculate_timing(self):
        """Calculate estimated print time."""
        # Write time: length / scan_speed
        self.metrics.write_time_sec = (
            self.metrics.total_write_length_um / self.scan_speed
        )
        
        # Z offset time
        self.metrics.z_offset_time_sec = (
            self.metrics.z_layer_count * self.Z_OFFSET_TIME_SEC
        )
        
        # Stage move time
        stage_travel_time = (
            self.metrics.stage_move_distance_um / self.STAGE_MOVE_SPEED_UM_PER_SEC
        )
        stage_settle_time = (
            self.metrics.stage_move_count * self.STAGE_SETTLE_TIME_SEC
        )
        self.metrics.stage_move_time_sec = stage_travel_time + stage_settle_time
        
        # Add base overhead
        self.metrics.overhead_time_sec += self.BASE_OVERHEAD_SEC
        
        # Total time
        self.metrics.estimated_time_sec = (
            self.metrics.write_time_sec +
            self.metrics.z_offset_time_sec +
            self.metrics.stage_move_time_sec +
            self.metrics.overhead_time_sec
        )


def evaluate_gwl(filepath: str, base_dir: str = None) -> Dict:
    """
    Evaluate a GWL file and return metrics.
    
    Args:
        filepath: Path to the GWL file
        base_dir: Base directory for resolving includes
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = GWLEvaluator(Path(base_dir) if base_dir else None)
    metrics = evaluator.evaluate_file(filepath)
    return metrics.to_dict()


def calibrate_evaluator():
    """
    Calibration helper - run on known files to tune constants.
    
    Targets:
    - meta_1.gwl: ~17 seconds
    - main_array.gwl (5x5): ~458 seconds
    """
    script_dir = Path(__file__).parent.parent
    example_dir = script_dir / "Outputs" / "Cylinder Example"
    
    print("="*60)
    print("GWL EVALUATOR CALIBRATION")
    print("="*60)
    
    # Evaluate meta_1.gwl
    meta_path = example_dir / "meta_1.gwl"
    if meta_path.exists():
        evaluator = GWLEvaluator(example_dir)
        metrics = evaluator.evaluate_file(str(meta_path))
        
        print(f"\nmeta_1.gwl:")
        print(f"  Write length: {metrics.total_write_length_um:.2f} um")
        print(f"  Write segments: {metrics.write_segment_count}")
        print(f"  Z layers: {metrics.z_layer_count}")
        print(f"  Estimated time: {metrics.estimated_time_sec:.2f} s (target: ~17 s)")
    
    # Evaluate main_array.gwl
    array_path = example_dir / "main_array.gwl"
    if array_path.exists():
        evaluator = GWLEvaluator(example_dir)
        metrics = evaluator.evaluate_file(str(array_path))
        
        print(f"\nmain_array.gwl (5x5):")
        print(f"  Write length: {metrics.total_write_length_um:.2f} um")
        print(f"  Write segments: {metrics.write_segment_count}")
        print(f"  Z layers: {metrics.z_layer_count}")
        print(f"  Stage moves: {metrics.stage_move_count}")
        print(f"  Stage distance: {metrics.stage_move_distance_um:.2f} um")
        print(f"  Estimated time: {metrics.estimated_time_sec:.2f} s (target: ~458 s)")
    
    print("="*60)


# ======================================================
# CLI
# ======================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python gwl_evaluator.py <file.gwl> [--calibrate]")
        print("       python gwl_evaluator.py --calibrate")
        sys.exit(1)
    
    if sys.argv[1] == '--calibrate':
        calibrate_evaluator()
    else:
        filepath = sys.argv[1]
        try:
            result = evaluate_gwl(filepath)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
