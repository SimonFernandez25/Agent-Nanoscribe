# ======================================================
# File: agent_test.py
# Integrated GWL Agent Test with Evaluation and Rendering
# ======================================================

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import tiktoken
import re

# Import evaluator and renderer
from gwl_evaluator import evaluate_gwl, GWLEvaluator
from gwl_renderer import render_gwl, GWLRenderer
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from gwl_validator import validate_gwl_files, validate_gwl_content

# ======================================================
# CONFIGURATION
# ======================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
PARENT_DIR = SCRIPT_DIR.parent
DOCS_DIR = PARENT_DIR / "Docs"
OUTPUTS_DIR = PARENT_DIR / "Outputs"
CYLINDER_EXAMPLE_DIR = OUTPUTS_DIR / "Cylinder Example"
API_FILE = DOCS_DIR / "API.txt"

OPENAI_API_KEY = API_FILE.read_text(encoding="utf-8").strip()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ======================================================
# MINIMAL CONTEXT (Cylinder Example + Langium Rules)
# ======================================================
def load_minimal_context():
    context = ""
    
    meta_gwl = CYLINDER_EXAMPLE_DIR / "meta_1.gwl"
    array_gwl = CYLINDER_EXAMPLE_DIR / "main_array.gwl"
    
    if meta_gwl.exists():
        context += f"\n### EXAMPLE META-ATOM GWL ###\n{meta_gwl.read_text(encoding='utf-8')}\n"
    if array_gwl.exists():
        context += f"\n### EXAMPLE ARRAY GWL ###\n{array_gwl.read_text(encoding='utf-8')}\n"
    
    context += """
### GWL GRAMMAR RULES ###
- VarDecl: var|local $name = value
- Assignment: set $name = value
- ForLoop: for $var = start to end [step val] ... end
- IfBlock: if condition ... [else ...] end
- IncludeStmt: include filepath
- CoordinateLine: expr expr expr [expr]
- CmdStmt: CommandName [args]
CRITICAL: Every variable must be declared with 'local' BEFORE use in loops.
"""
    return context

# ======================================================
# SYSTEM PROMPTS (from simple_gwl_agent.py)
# ======================================================
GEOMETRY_PROMPT = """You are a geometry planner for Nanoscribe metasurfaces.
Output ONE structured JSON object describing:
- 'array': {'rows': int, 'cols': int, 'spacing_um': float}
- 'meta_atoms': [{'id': int, 'shape': str, dimensions...}]
- 'pattern_map': 2D list of IDs

Output valid JSON only. No commentary."""

META_ATOM_PROMPT = """You are a DescribeX GWL generator for the Nanoscribe Quantum X.
Produce ONLY valid GWL code for a single meta-atom.

CRITICAL RULES:
- Every variable used in a loop must be declared first with 'local'.
- Center at origin (X=0, Y=0), start at ZOffset 0, build upward.
- Use 'end' for blocks. Use 'set' for assignments inside loops.
- Output only GWL code, no commentary."""

ARRAY_PROMPT = """You are a DescribeX GWL composition agent for the Nanoscribe Quantum X.
Build a valid master .GWL file that arranges meta-atom GWLs into an array.

CRITICAL RULES:
- Every variable must be declared with 'local' BEFORE loops.
- Use MoveStageX/MoveStageY for positioning.
- Output ONLY valid GWL code, no commentary."""

DEBUG_PROMPT = """You are a GWL debugger. Fix the code based on validation errors.
Output ONLY the corrected GWL code."""

LANGIUM_RULES = """
GWL GRAMMAR:
- VarDecl: var|local $name = value
- Assignment: set $name = value
- ForLoop: for $var = start to end [step val] ... end
- IfBlock: if condition ... end
- Include: include filepath
"""

# ======================================================
# HELPERS
# ======================================================
def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def clean_code(code: str) -> str:
    if code.startswith("```"):
        code = re.sub(r'^```\w*\n?', '', code)
        code = re.sub(r'\n?```$', '', code)
    return code.strip()

def save_json(path: Path, filename: str, content):
    with open(path / filename, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)

def format_errors(validation) -> str:
    errors = ""
    for v in validation.validations:
        if not v.valid:
            for err in v.errors:
                errors += f"Line {err.line}: {err.message}\n"
    return errors

def debug_code(code: str, errors: str, llm, stats: Dict) -> tuple:
    human = f"GWL CODE:\n{code}\n\nERRORS:\n{errors}\n\nGRAMMAR:\n{LANGIUM_RULES}\n\nFix errors."
    msgs = [SystemMessage(content=DEBUG_PROMPT), HumanMessage(content=human)]
    stats["input_tokens"] += count_tokens(DEBUG_PROMPT) + count_tokens(human)
    result = llm.invoke(msgs)
    fixed = clean_code(result.content)
    stats["output_tokens"] += count_tokens(fixed)
    stats["debug_calls"] += 1
    return fixed, stats


# ======================================================
# RUN SINGLE AGENT
# ======================================================
def run_agent(prompt: str, run_dir: Path, run_num: int) -> Dict:
    """Run the agent pipeline once and return metrics."""
    
    print(f"\n{'='*50}")
    print(f"RUN {run_num}")
    print(f"{'='*50}")
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    context = load_minimal_context()
    
    stats = {
        "input_tokens": 0,
        "output_tokens": 0,
        "llm_calls": 0,
        "debug_calls": 0,
        "meta_debugs": 0,
        "array_debugs": 0
    }
    
    start_time = time.time()
    
    # 1. Geometry
    print("  [1/3] Parsing geometry...")
    human = f"{prompt}\n\nReference:\n{context}"
    msgs = [SystemMessage(content=GEOMETRY_PROMPT), HumanMessage(content=human)]
    stats["input_tokens"] += count_tokens(GEOMETRY_PROMPT) + count_tokens(human)
    
    result = llm.invoke(msgs)
    geom_json = clean_code(result.content)
    stats["output_tokens"] += count_tokens(geom_json)
    stats["llm_calls"] += 1
    
    save_json(run_dir, "geometry.json", json.loads(geom_json))
    geom = json.loads(geom_json)
    
    # 2. Meta-atom GWL
    print("  [2/3] Generating meta-atom GWL...")
    gwl_files = []
    for atom in geom.get("meta_atoms", []):
        name = f"meta_{atom['id']}"
        human = f"META-ATOM: {json.dumps(atom)}\n\nREFERENCE:\n{context}"
        msgs = [SystemMessage(content=META_ATOM_PROMPT), HumanMessage(content=human)]
        stats["input_tokens"] += count_tokens(META_ATOM_PROMPT) + count_tokens(human)
        
        result = llm.invoke(msgs)
        code = clean_code(result.content)
        stats["output_tokens"] += count_tokens(code)
        stats["llm_calls"] += 1
        
        # Debug loop (max 2)
        for _ in range(2):
            val = validate_gwl_content({name: code})
            if val.valid:
                break
            stats["meta_debugs"] += 1
            code, stats = debug_code(code, format_errors(val), llm, stats)
        
        fpath = run_dir / f"{name}.gwl"
        fpath.write_text(code, encoding="utf-8")
        gwl_files.append(str(fpath))
    
    # 3. Array GWL
    print("  [3/3] Generating array GWL...")
    files = "\n".join([f"meta_{a['id']}.gwl" for a in geom.get("meta_atoms", [])])
    human = f"ARRAY: {json.dumps(geom.get('array', {}))}\nFILES: {files}\n\nREFERENCE:\n{context}"
    msgs = [SystemMessage(content=ARRAY_PROMPT), HumanMessage(content=human)]
    stats["input_tokens"] += count_tokens(ARRAY_PROMPT) + count_tokens(human)
    
    result = llm.invoke(msgs)
    code = clean_code(result.content)
    stats["output_tokens"] += count_tokens(code)
    stats["llm_calls"] += 1
    
    for _ in range(2):
        val = validate_gwl_content({"main_array.gwl": code})
        if val.valid:
            break
        stats["array_debugs"] += 1
        code, stats = debug_code(code, format_errors(val), llm, stats)
    
    array_path = run_dir / "main_array.gwl"
    array_path.write_text(code, encoding="utf-8")
    gwl_files.append(str(array_path))
    
    total_time = time.time() - start_time
    
    # 4. Validation
    print("  Validating...")
    validation = validate_gwl_files(gwl_files)
    total_errors = sum(len(v.errors) for v in validation.validations)
    
    # 5. Evaluation (manufacturing metrics)
    print("  Evaluating manufacturing metrics...")
    eval_metrics = {}
    for fpath in gwl_files:
        try:
            evaluator = GWLEvaluator(run_dir)
            m = evaluator.evaluate_file(fpath)
            eval_metrics[Path(fpath).name] = m.to_dict()
        except Exception as e:
            eval_metrics[Path(fpath).name] = {"error": str(e)}
    
    # 6. Rendering
    print("  Rendering structure views...")
    render_paths = {}
    for fpath in gwl_files:
        try:
            top, side = render_gwl(fpath, str(run_dir))
            render_paths[Path(fpath).name] = {"top": top, "side": side}
        except Exception as e:
            render_paths[Path(fpath).name] = {"error": str(e)}
    
    # Compile results
    results = {
        "run": run_num,
        "time_sec": round(total_time, 2),
        "input_tokens": stats["input_tokens"],
        "output_tokens": stats["output_tokens"],
        "total_tokens": stats["input_tokens"] + stats["output_tokens"],
        "llm_calls": stats["llm_calls"],
        "debug_calls": stats["debug_calls"],
        "meta_debugs": stats["meta_debugs"],
        "array_debugs": stats["array_debugs"],
        "valid": validation.valid,
        "total_errors": total_errors,
        "manufacturing": eval_metrics,
        "renders": render_paths
    }
    
    save_json(run_dir, "results.json", results)
    
    # Print summary
    status = "[OK]" if validation.valid else f"[ERR:{total_errors}]"
    print(f"  {status} | {total_time:.1f}s | {stats['input_tokens'] + stats['output_tokens']} tok | debugs: {stats['debug_calls']}")
    
    if eval_metrics.get("meta_1.gwl"):
        m = eval_metrics["meta_1.gwl"]
        print(f"  Meta: {m.get('estimated_time_sec', 0):.1f}s est, {m.get('write_segment_count', 0)} segments")
    
    if eval_metrics.get("main_array.gwl"):
        m = eval_metrics["main_array.gwl"]
        print(f"  Array: {m.get('estimated_time_sec', 0):.1f}s est, {m.get('write_segment_count', 0)} segments")
    
    return results


# ======================================================
# MAIN
# ======================================================
def main():
    print("\n" + "="*60)
    print("GWL AGENT INTEGRATED TEST")
    print("="*60)
    
    # Load prompt
    prompt_file = PARENT_DIR / "prompt.txt"
    prompt = prompt_file.read_text(encoding="utf-8").strip().strip('"')
    print(f"Prompt: {prompt[:80]}...")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = SCRIPT_DIR / f"test_{timestamp}"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Run 3 times
    for run_num in range(1, 4):
        run_dir = test_dir / f"run_{run_num}"
        try:
            results = run_agent(prompt, run_dir, run_num)
            all_results.append(results)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for r in all_results:
        status = "[OK]" if r.get("valid") else f"[ERR:{r.get('total_errors', '?')}]"
        print(f"Run {r['run']}: {status} | {r['time_sec']}s | {r['total_tokens']} tok | debugs: {r['debug_calls']}")
    
    save_json(test_dir, "all_results.json", all_results)
    
    print(f"\nResults saved to: {test_dir}")
    print("="*60)
    
    return test_dir


if __name__ == "__main__":
    main()
