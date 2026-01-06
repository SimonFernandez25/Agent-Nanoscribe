# ======================================================
# File: gwl_renderer.py
# GWL Structure Renderer - Top-down and Side Views
# ======================================================

import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import numpy as np


@dataclass
class RenderData:
    """Data collected for rendering."""
    polylines: List[List[Tuple[float, float, float]]] = field(default_factory=list)
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    def add_polyline(self, points: List[Tuple[float, float, float]]):
        if len(points) >= 2:
            self.polylines.append(points)
    
    def calculate_bounds(self):
        if not self.polylines:
            self.bounds = {'x': (-1, 1), 'y': (-1, 1), 'z': (0, 1)}
            return
        
        all_points = [p for poly in self.polylines for p in poly]
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        zs = [p[2] for p in all_points]
        
        margin = 0.1
        x_range = max(xs) - min(xs) or 1
        y_range = max(ys) - min(ys) or 1
        z_range = max(zs) - min(zs) or 1
        
        self.bounds = {
            'x': (min(xs) - margin * x_range, max(xs) + margin * x_range),
            'y': (min(ys) - margin * y_range, max(ys) + margin * y_range),
            'z': (min(zs) - margin * z_range, max(zs) + margin * z_range)
        }


class GWLRenderer:
    """
    Renders GWL toolpaths as 2D projections.
    Produces top-down (X-Y) and side (X-Z) views.
    """
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path.cwd()
        self.variables: Dict[str, float] = {}
        self.current_z_offset: float = 0.0
        self.pending_coords: List[Tuple[float, float, float]] = []
        self.render_data = RenderData()
        
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
        }
    
    def render_file(self, filepath: str, output_dir: str = None) -> Tuple[str, str]:
        """
        Render a GWL file and save top-down and side view images.
        Returns paths to the generated images.
        """
        path = Path(filepath)
        if not path.is_absolute():
            path = self.base_dir / path
        
        if not path.exists():
            raise FileNotFoundError(f"GWL file not found: {path}")
        
        self.base_dir = path.parent
        content = path.read_text(encoding='utf-8')
        
        # Reset state
        self._reset()
        
        # Parse and collect render data
        self._execute(content)
        self.render_data.calculate_bounds()
        
        # Generate plots
        output_path = Path(output_dir) if output_dir else path.parent
        stem = path.stem
        
        top_view_path = output_path / f"{stem}_top_view.png"
        side_view_path = output_path / f"{stem}_side_view.png"
        
        self._render_top_view(top_view_path)
        self._render_side_view(side_view_path)
        
        return str(top_view_path), str(side_view_path)
    
    def _reset(self):
        self.variables = {}
        self.current_z_offset = 0.0
        self.pending_coords = []
        self.render_data = RenderData()
    
    def _execute(self, content: str):
        lines = content.split('\n')
        self._execute_lines(lines)
    
    def _execute_lines(self, lines: List[str]):
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line or line.startswith('%'):
                i += 1
                continue
            
            if '%' in line:
                line = line[:line.index('%')].strip()
                if not line:
                    i += 1
                    continue
            
            if line.startswith('var ') or line.startswith('local '):
                self._handle_var_decl(line)
            elif line.startswith('set '):
                self._handle_assignment(line)
            elif line.startswith('for '):
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
                pass
            elif self._is_coordinate_line(line):
                self._handle_coordinate(line)
            else:
                self._handle_command(line)
            
            i += 1
    
    def _find_matching_end(self, lines: List[str], start: int) -> int:
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
        match = re.match(r'(?:var|local)\s+\$(\w+)\s*=\s*(.+)', line)
        if match:
            self.variables[match.group(1)] = self._eval_expr(match.group(2))
    
    def _handle_assignment(self, line: str):
        match = re.match(r'set\s+\$(\w+)\s*=\s*(.+)', line)
        if match:
            self.variables[match.group(1)] = self._eval_expr(match.group(2))
    
    def _handle_for_loop(self, header: str, body: List[str]):
        match = re.match(r'for\s+\$(\w+)\s*=\s*(.+?)\s+to\s+(.+?)(?:\s+step\s+(.+))?$', header)
        if not match:
            return
        
        var_name = match.group(1)
        start_val = self._eval_expr(match.group(2))
        end_val = self._eval_expr(match.group(3))
        step_val = self._eval_expr(match.group(4)) if match.group(4) else 1.0
        
        if step_val == 0:
            return
        
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
        match = re.match(r'if\s+(.+)', header)
        if not match:
            return
        
        condition = match.group(1)
        
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
        match = re.match(r'[Ii]nclude\s+(.+)', line)
        if match:
            filepath = match.group(1).strip()
            full_path = self.base_dir / filepath
            if full_path.exists():
                content = full_path.read_text(encoding='utf-8')
                parent_vars = self.variables.copy()
                parent_z = self.current_z_offset
                parent_pending = self.pending_coords.copy()
                
                self.current_z_offset = 0.0
                self.pending_coords = []
                self._execute(content)
                
                self.variables = parent_vars
                self.current_z_offset = parent_z
                self.pending_coords = parent_pending
    
    def _handle_command(self, line: str):
        parts = line.split()
        if not parts:
            return
        
        cmd = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd == 'AddZOffset' and args:
            self.current_z_offset += self._eval_expr(args[0])
        elif cmd == 'ZOffset' and args:
            self.current_z_offset = self._eval_expr(args[0])
    
    def _is_coordinate_line(self, line: str) -> bool:
        parts = line.split()
        if len(parts) < 3 or len(parts) > 4:
            return False
        first = parts[0]
        if first[0].isalpha() and not first.startswith('$'):
            return False
        return True
    
    def _handle_coordinate(self, line: str):
        parts = line.split()
        if len(parts) >= 3:
            x = self._eval_expr(parts[0])
            y = self._eval_expr(parts[1])
            z = self._eval_expr(parts[2]) + self.current_z_offset
            self.pending_coords.append((x, y, z))
    
    def _handle_write(self):
        if len(self.pending_coords) >= 2:
            self.render_data.add_polyline(self.pending_coords.copy())
        self.pending_coords = []
    
    def _eval_expr(self, expr: str) -> float:
        if not expr:
            return 0.0
        
        expr = expr.strip()
        
        try:
            return float(expr)
        except ValueError:
            pass
        
        if expr.startswith('$') and re.match(r'^\$\w+$', expr):
            return self.variables.get(expr[1:], 0.0)
        
        if expr.startswith('-$') and re.match(r'^-\$\w+$', expr):
            return -self.variables.get(expr[2:], 0.0)
        
        def replace_var(match):
            return f'({self.variables.get(match.group(1), 0.0)})'
        
        expr = re.sub(r'\$(\w+)', replace_var, expr)
        
        # Handle functions with balanced parens
        for _ in range(20):
            found = False
            for func_name, func in self.functions.items():
                pattern = rf'\b{func_name}\s*\('
                match = re.search(pattern, expr)
                if match:
                    start = match.end() - 1
                    depth, end = 1, start + 1
                    while end < len(expr) and depth > 0:
                        if expr[end] == '(':
                            depth += 1
                        elif expr[end] == ')':
                            depth -= 1
                        end += 1
                    if depth == 0:
                        found = True
                        try:
                            inner = eval(expr[start+1:end-1], {"__builtins__": {}}, {})
                            result = func(inner)
                            expr = expr[:match.start()] + str(result) + expr[end:]
                        except:
                            expr = expr[:match.start()] + '0' + expr[end:]
                        break
            if not found:
                break
        
        try:
            return float(eval(expr, {"__builtins__": {}}, {}))
        except:
            return 0.0
    
    def _eval_condition(self, condition: str) -> bool:
        for op in ['==', '!=', '<=', '>=', '<', '>']:
            if op in condition:
                parts = condition.split(op, 1)
                if len(parts) == 2:
                    left, right = self._eval_expr(parts[0]), self._eval_expr(parts[1])
                    if op == '==': return left == right
                    if op == '!=': return left != right
                    if op == '<=': return left <= right
                    if op == '>=': return left >= right
                    if op == '<': return left < right
                    if op == '>': return left > right
        return False
    
    def _render_top_view(self, output_path: Path):
        """Render X-Y top-down view."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        bounds = self.render_data.bounds
        
        # Draw polylines
        for poly in self.render_data.polylines:
            # Color by Z height
            z_vals = [p[2] for p in poly]
            avg_z = sum(z_vals) / len(z_vals)
            z_range = bounds['z'][1] - bounds['z'][0] or 1
            z_norm = (avg_z - bounds['z'][0]) / z_range
            color = plt.cm.viridis(z_norm)
            
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            ax.plot(xs, ys, color=color, linewidth=0.5, alpha=0.7)
        
        ax.set_xlim(bounds['x'])
        ax.set_ylim(bounds['y'])
        ax.set_aspect('equal')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title('Top View (X-Y)')
        
        # Add scale bar
        self._add_scale_bar(ax, bounds['x'], bounds['y'])
        
        # Colorbar for Z
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(bounds['z'][0], bounds['z'][1]))
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label('Z (μm)')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _render_side_view(self, output_path: Path):
        """Render X-Z side view."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bounds = self.render_data.bounds
        
        for poly in self.render_data.polylines:
            xs = [p[0] for p in poly]
            zs = [p[2] for p in poly]
            ax.plot(xs, zs, color='#2196F3', linewidth=0.5, alpha=0.5)
        
        ax.set_xlim(bounds['x'])
        ax.set_ylim(bounds['z'])
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Z (μm)')
        ax.set_title('Side View (X-Z)')
        
        self._add_scale_bar(ax, bounds['x'], bounds['z'])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _add_scale_bar(self, ax, x_bounds, y_bounds):
        """Add a measurement scale bar to the plot."""
        x_range = x_bounds[1] - x_bounds[0]
        
        # Choose nice scale bar length
        scale_options = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
        target = x_range * 0.2
        scale_len = min(scale_options, key=lambda x: abs(x - target))
        
        # Position in lower right
        x_pos = x_bounds[1] - x_range * 0.15
        y_pos = y_bounds[0] + (y_bounds[1] - y_bounds[0]) * 0.05
        
        ax.plot([x_pos - scale_len, x_pos], [y_pos, y_pos], 'k-', linewidth=2)
        ax.text(x_pos - scale_len/2, y_pos + (y_bounds[1] - y_bounds[0]) * 0.02,
                f'{scale_len} μm', ha='center', fontsize=8)


def render_gwl(filepath: str, output_dir: str = None) -> Tuple[str, str]:
    """
    Render a GWL file and return paths to top and side view images.
    """
    renderer = GWLRenderer()
    return renderer.render_file(filepath, output_dir)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python gwl_renderer.py <file.gwl> [output_dir]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        top, side = render_gwl(filepath, output_dir)
        print(f"Top view: {top}")
        print(f"Side view: {side}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
