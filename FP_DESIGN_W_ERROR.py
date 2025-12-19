# ======================================================
# File: FP_DESIGN_W_ERROR.py
# ======================================================

import os, time, uuid, json, hashlib, fitz, re
from pathlib import Path
from typing import TypedDict, Optional, List, Dict
from datetime import datetime
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ======================================================
# STATE
# ======================================================
class AgentState(TypedDict, total=False):
    user_prompt: str
    project_name: str
    project_dir: Path
    geometry_json: str
    geometry_metadata: Dict
    design_intent: str
    design_intent_metadata: Dict
    meta_gwls: List[str]
    meta_gwls_metadata: List[Dict]
    master_gwl: str
    master_gwl_metadata: Dict
    errors: Optional[str]

# ======================================================
# CONFIGURATION
# ======================================================
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# Get parent directory (Agentic Design folder)
BASE_DIR = SCRIPT_DIR.parent

# Set paths relative to BASE_DIR
DOCS_DIR = BASE_DIR / "Docs"
OUTPUTS_BASE_DIR = SCRIPT_DIR / "Outputs"
API_FILE = DOCS_DIR / "API.txt"

OPENAI_API_KEY = API_FILE.read_text(encoding="utf-8").strip()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ======================================================
# HELPERS
# ======================================================
def extract_text(file: Path):
    """Safely extract text from PDF or TXT, skip everything else."""
    if not file.exists():
        return ""

    try:
        ext = file.suffix.lower()

        # ----- PDF -----
        if ext == ".pdf":
            text = ""
            with fitz.open(file) as doc:
                for page in doc:
                    text += page.get_text()
            return text[:8000]

        # ----- TXT -----
        elif ext == ".txt":
            with open(file, "rb") as f:
                raw = f.read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1", errors="ignore")
            return text[:8000]

        # ----- Unsupported -----
        else:
            print(f"Skipping unsupported file: {file.name}")
            return ""

    except Exception as e:
        print(f"[WARN] Failed to read {file.name}: {e}")
        return ""


def load_docs_for_context():
    """Load all valid Docs files and mark their boundaries."""
    context_blobs = []
    for p in sorted(DOCS_DIR.glob("*")):
        text = extract_text(p)
        if text:
            tagged = f"\n### BEGIN {p.name.upper()} ###\n{text}\n### END {p.name.upper()} ###"
            context_blobs.append(tagged)
            print(f"Loaded {p.name} ({len(text)} chars)")
    return "\n".join(context_blobs)


def generate_project_name_from_prompt(prompt: str) -> str:
    """
    Generate a simple project name from timestamp.
    No LLM call needed - saves tokens and API overhead.
    """
    # Use timestamp-based naming for efficiency
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract first few meaningful words from prompt for readability
    words = re.findall(r'\w+', prompt.lower())
    # Take first 3 words, skip common words
    skip_words = {'a', 'an', 'the', 'of', 'with', 'for', 'to', 'from', 'in', 'on', 'at'}
    meaningful = [w for w in words if w not in skip_words][:3]
    prefix = '_'.join(meaningful) if meaningful else 'design'
    
    return f"{prefix}_{timestamp}"


def create_project_structure(project_name: str) -> Path:
    """Create organized project directory structure."""
    project_dir = OUTPUTS_BASE_DIR / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def save_json_output(project_dir: Path, filename: str, content: any, node_name: str, metadata: Dict = None) -> Dict:
    """Save structured JSON output with metadata."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "node_name": node_name,
        "content": content,
        "metadata": metadata or {}
    }
    
    json_path = project_dir / filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    return output


def generate_pdf_report(project_dir: Path, state: AgentState, metrics: Dict):
    """Generate comprehensive PDF report of the agent run."""
    pdf_path = project_dir / "project_report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=8,
        leftIndent=20,
        rightIndent=20,
        spaceAfter=12
    )
    
    # Cover Page
    story.append(Paragraph("Nanoscribe Design Agent Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"<b>Project:</b> {state['project_name']}", styles['Normal']))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    
    # User Prompt
    story.append(Paragraph("User Prompt", heading_style))
    story.append(Paragraph(state['user_prompt'], styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Design Intent
    story.append(Paragraph("Design Intent", heading_style))
    story.append(Paragraph(state.get('design_intent', 'N/A'), styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Geometry Specification
    story.append(Paragraph("Geometry Specification", heading_style))
    geom_json = json.loads(state.get('geometry_json', '{}'))
    geom_formatted = json.dumps(geom_json, indent=2)
    story.append(Preformatted(geom_formatted, code_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Meta-Atoms Summary
    story.append(Paragraph("Generated Meta-Atoms", heading_style))
    meta_atoms = geom_json.get('meta_atoms', [])
    if meta_atoms:
        table_data = [['ID', 'Shape', 'Properties']]
        for atom in meta_atoms:
            atom_id = str(atom.get('id', 'N/A'))
            shape = atom.get('shape', 'N/A')
            props = ', '.join([f"{k}={v}" for k, v in atom.items() if k not in ['id', 'shape']])
            table_data.append([atom_id, shape, props[:50]])
        
        table = Table(table_data, colWidths=[0.5*inch, 1*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
    story.append(Spacer(1, 0.3*inch))
    
    # GWL Files Generated
    story.append(Paragraph("GWL Files Generated", heading_style))
    meta_gwls = state.get('meta_gwls', [])
    for gwl_path in meta_gwls:
        gwl_file = Path(gwl_path)
        if gwl_file.exists():
            story.append(Paragraph(f"<b>{gwl_file.name}</b>", styles['Normal']))
            gwl_content = gwl_file.read_text(encoding='utf-8')[:500]
            story.append(Preformatted(gwl_content + "\n...", code_style))
    
    master_gwl = state.get('master_gwl', '')
    if master_gwl and Path(master_gwl).exists():
        story.append(Paragraph("<b>main_array.gwl</b>", styles['Normal']))
        master_content = Path(master_gwl).read_text(encoding='utf-8')[:500]
        story.append(Preformatted(master_content + "\n...", code_style))
    
    story.append(PageBreak())
    
    # Statistics
    story.append(Paragraph("Run Statistics", heading_style))
    stats_data = [
        ['Metric', 'Value'],
        ['Total Runtime', f"{metrics.get('total_runtime_sec', 0):.2f} seconds"],
        ['Input Tokens', str(metrics.get('input_tokens', 0))],
        ['Output Tokens', str(metrics.get('output_tokens', 0))],
        ['Total Tokens', str(metrics.get('total_tokens', 0))],
        ['Meta-Atoms Generated', str(len(meta_gwls))],
        ['GWL Files Created', str(len(meta_gwls) + 1)]
    ]
    
    stats_table = Table(stats_data, colWidths=[2.5*inch, 2.5*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(stats_table)
    
    # Build PDF
    doc.build(story)
    print(f"\n✓ PDF Report generated: {pdf_path}")


# ======================================================
# NODE 1: GEOMETRY
# ======================================================
def node_geometry(state):
    """
    Planner node: outputs structured JSON with array, meta_atoms, and pattern_map.
    Each unique meta-atom is assigned a numeric ID (1, 2, 3, ...).
    The pattern_map defines which meta-atom appears at each (row, col).
    """
    llm = ChatOpenAI(
    model="gpt-5o-mini",
    temperature=0.1,
)

    prompt = state["user_prompt"]
    project_dir = state["project_dir"]
    docs_text = load_docs_for_context()

    system = (
        "You are a geometry planner for Nanoscribe metasurfaces.\n"
        "Your task is to output ONE structured JSON object describing:\n"
        "- 'array': global layout with {'rows': int, 'cols': int, 'spacing_um': float}\n"
        "- 'meta_atoms': list of UNIQUE meta-atoms with {'id': int, 'shape': str, dimensions...}\n"
        "- 'pattern_map': 2D list (rows×cols) of integer IDs referencing meta_atoms.\n\n"
        "Rules:\n"
        "- Assign IDs sequentially starting from 1.\n"
        "- Each unique meta-atom geometry should appear only once in 'meta_atoms'.\n"
        "- pattern_map should only contain those numeric IDs.\n"
        "- If the array is uniform, all entries in pattern_map should be [1].\n"
        "- Output valid JSON only. Do NOT include commentary or example text.\n"
        "Example output:\n"
        "{\n"
        "  \"array\": {\"rows\": 3, \"cols\": 3, \"spacing_um\": 5.0},\n"
        "  \"meta_atoms\": [\n"
        "    {\"id\": 1, \"shape\": \"cylinder\", \"height_um\": 1.0, \"radius_um\": 1.0},\n"
        "    {\"id\": 2, \"shape\": \"cube\", \"height_um\": 1.0, \"width_um\": 1.0}\n"
        "  ],\n"
        "  \"pattern_map\": [\n"
        "    [1, 2, 1],\n"
        "    [2, 1, 2],\n"
        "    [1, 2, 1]\n"
        "  ]\n"
        "}"
    )

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"{prompt}\n\nReference:\n{docs_text}")
    ]

    start_time = time.time()
    result = llm.invoke(messages)
    elapsed = time.time() - start_time

    geometry_json = result.content.strip()

    # Save structured JSON output
    metadata = {
        "elapsed_time_sec": elapsed,
        "model": "gpt-5o-mini"
    }
    save_json_output(project_dir, "geometry_output.json", json.loads(geometry_json), "geometry", metadata)

    print("\n=== GEOMETRY PLANNER OUTPUT ===")
    print(geometry_json)
    print(f"  Time taken: {elapsed:.2f}s")
    print("================================\n")

    state["geometry_json"] = geometry_json
    state["geometry_metadata"] = metadata
    return state

# ======================================================
# NODE 2: DESIGN INTENT
# ======================================================
def node_design_intent(state):
    llm = ChatOpenAI(
    model="gpt-5o-mini",
    temperature=0.1,
)
    prompt = state["user_prompt"]
    project_dir = state["project_dir"]
    
    system = (
        "Summarize the user's design goal in one concise paragraph. "
        "Focus on geometry, symmetry, or optical/mechanical intent."
    )
    messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
    
    start_time = time.time()
    result = llm.invoke(messages)
    elapsed = time.time() - start_time
    
    design_intent = result.content.strip()
    
    # Save both JSON and text versions
    metadata = {
        "elapsed_time_sec": elapsed,
        "model": "gpt-5o-mini"
    }
    save_json_output(project_dir, "design_intent.json", design_intent, "design_intent", metadata)
    (project_dir / "design_intent.txt").write_text(design_intent, encoding="utf-8")
    
    print("\n=== DESIGN INTENT NODE OUTPUT ===\n", design_intent, "\n=================================\n")
    
    state["design_intent"] = design_intent
    state["design_intent_metadata"] = metadata
    return state

# ======================================================
# NODE 3: META-ATOM GWL GENERATOR
# ======================================================
def node_generate_meta_gwls(state):
    """
    Generates ONE GWL per unique meta-atom, using only its own geometry JSON.
    The array-level data (rows, cols, spacing, pattern_map) is never exposed.
    Each atom is treated as fully independent and unaware of any array context.
    """
    llm = ChatOpenAI(
    model="gpt-5o-mini",
    temperature=0.1,
)
    geom_data = json.loads(state["geometry_json"])
    project_dir = state["project_dir"]

    meta_atoms = geom_data.get("meta_atoms", [])
    if not meta_atoms:
        print("[WARN] No meta_atoms found in geometry JSON.")
        return state

    # --- Load Docs (good + bad) for syntax context only ---
    context_blobs = []
    for p in sorted(DOCS_DIR.glob("*.txt")):
        text = extract_text(p)
        if not text:
            continue
        if "Generated__" in p.name or "Debug" in p.name or "Error" in p.name:
            tagged = f"\n### BAD EXAMPLE ({p.name}) ###\n{text[:4000]}\n### END BAD EXAMPLE ###"
        else:
            tagged = f"\n### REF DOC ({p.name}) ###\n{text[:4000]}\n### END REF DOC ###"
        context_blobs.append(tagged)
        print(f"Loaded {p.name} ({len(text)} chars)")
    docs_text = "\n".join(context_blobs)

    gwl_files = []
    gwl_metadata_list = []

    # === Generate each unique meta-atom in isolation ===
    for atom in meta_atoms:
        atom_id = atom.get("id", 1)
        atom_name = f"meta_{atom_id}"
        atom_prompt = json.dumps(atom, indent=2)

        # --- System role: strictly single meta-atom mode ---
        system = (
            "You are a DescribeX GWL generator for the Nanoscribe Quantum X.\n"
            "Produce ONLY valid GWL code for a single Unique meta-atom using official syntax.\n"
            "Every variable used in a loop or expression must be declared first with 'local'. THIS IS A MUST! DO NOT assume that just because a variable is used in a loop or expression, it has been declared. You must declare every variable used in a loop or expression."
            "Center geometry at the origin (X = 0 Y = 0), start writing at ZOffset 0 "
            "(flush to substrate), and build upward.\n"
            "Declare all variables with 'local' or 'var'. Use consistent indentation "
            "and proper loop termination ('end').\n"
            "Output only GWL code—no explanations or comments beyond standard block headers."
            
    "IMPORTANT RULES FOR VALID GWL:\n"
    "- Every coordinate line must contain only explicit variables or numbers.\n"
    "- Do NOT write `-$x` or expressions inline — instead define them with `set` first.\n"
    "- For example, instead of `-$xHalf $y 0`, do:\n"
    "    set $x1 = -$xHalf\n"
    "    set $x2 =  $xHalf\n"
    "    set $y1 = $y\n"
    "    set $y2 = $y\n"
    "    $x1 $y1 0\n"
    "    $x2 $y2 0\n"
    "    Write\n"
)


        # --- Human message: geometry + syntax references only ---
        human = f"""
META-ATOM SPECIFICATION (JSON):
{atom_prompt}

REFERENCE & BAD EXAMPLES (trimmed):
{docs_text}

CORRECT FORMAT MODEL:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example Meta-Atom (Cube)
% Objective: 63x
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
var $objective = 63
local $height = 2.0
local $voxel_lateral = 0.1
local $axial_step = 0.3
local $axial_overlap = 0.5
local $dz = $axial_step * (1 - $axial_overlap)
local $slices = ceil($height / $dz)
local $hatch = $voxel_lateral

XOffset 0
YOffset 0
ZOffset 0
PowerScaling 0.6 ****Should always be this value
LaserPower 40
ScanSpeed 100
LineNumber 1

local $slice = 0
local $y = 0
for $slice = 0 to $slices - 1 ***The slice was defined before calling in a loop!!!!
    if $slice > 0
        AddZOffset $dz
    end
    for $y = -1 to 1 step $hatch
        ...
    end
end

\tif $slice > 0
\t\tAddZOffset $dz
\tend
\tfor $y = -1 to 1 step $hatch
\t\tset $x1 = -1
\t\tset $x2 =  1
\t\tset $y1 =  $y
\t\tset $y2 =  $y
\t\t$x1 $y1 0
\t\t$x2 $y2 0
\t\tWrite
\tend
end

ZOffset 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Generate the meta-atom's GWL code following this structure and DescribeX rules.
"""

        start_time = time.time()
        messages = [SystemMessage(content=system), HumanMessage(content=human)]
        result = llm.invoke(messages)
        elapsed = time.time() - start_time
        
        gwl_code = result.content.strip()

        fpath = project_dir / f"{atom_name}.gwl"
        fpath.write_text(gwl_code, encoding="utf-8")
        gwl_files.append(str(fpath))
        
        # Save metadata
        gwl_meta = {
            "atom_id": atom_id,
            "filename": f"{atom_name}.gwl",
            "elapsed_time_sec": elapsed,
            "file_size_bytes": len(gwl_code),
            "atom_spec": atom
        }
        save_json_output(project_dir, f"{atom_name}.json", gwl_code, "meta_gwl_generator", gwl_meta)
        gwl_metadata_list.append(gwl_meta)
        
        print(f"Generated {fpath.name} ({len(gwl_code)} chars)")

    state["meta_gwls"] = gwl_files
    state["meta_gwls_metadata"] = gwl_metadata_list
    return state

def node_build_master_gwl(state):
    """
    Agent node: builds a DescribeX-compatible master array GWL
    using the geometry JSON, meta-atom GWLs, and user prompt.
    Incorporates 'Generated__' and debug cache .txt files as
    examples of incorrect syntax to avoid.
    """

    llm = ChatOpenAI(
    model="gpt-5o-mini",
    temperature=0.1,
)

    project_dir = state["project_dir"]

    # === Load Docs and Debug Caches ===
    context_blobs = []
    for p in sorted(DOCS_DIR.glob("*.txt")):
        text = extract_text(p)
        if not text:
            continue

        # Label debug files as BAD examples
        if "Generated__" in p.name or "Debug" in p.name or "Error" in p.name:
            tagged = f"\n### BAD EXAMPLE ({p.name}) ###\n{text[:4000]}\n### END BAD EXAMPLE ###"
        else:
            tagged = f"\n### REF DOC ({p.name}) ###\n{text[:4000]}\n### END REF DOC ###"

        context_blobs.append(tagged)
        print(f"Loaded {p.name} ({len(text)} chars)")

    docs_text = "\n".join(context_blobs)

    # === Retrieve Geometry & Meta Info ===
    geom_data = json.loads(state["geometry_json"])
    user_prompt = state.get("user_prompt", "")
    meta_files = state.get("meta_gwls", [])

    rows, cols = geom_data.get("array_dims", [1, 1])
    spacing = geom_data.get("spacing_um", 5.0)
    pattern_map = geom_data.get("pattern_map", [])
    meta_atoms = geom_data.get("meta_atoms", [])
    size = (
        meta_atoms[0].get("diameter_um")
        or meta_atoms[0].get("width_um")
        or meta_atoms[0].get("size_um")
        or 1.0
    )

    include_snippet = "\n".join([f"meta_{a['id']}.gwl" for a in meta_atoms])
    pattern_text = json.dumps(pattern_map, indent=2)

    # === SYSTEM MESSAGE ===
    system = (
        "You are a DescribeX GWL composition agent for the Nanoscribe Quantum X.\n"
        "Your task: build a valid master .GWL file that arranges previously generated meta-atom GWLs "
        "into an array according to the geometry JSON and user design prompt.\n\n"
        "Follow ONLY the official DescribeX command syntax shown in reference docs (MoveStageX, MoveStageY, Include, etc.).\n"
        "Center the array around the origin, keep all meta-atoms normal to the substrate, and ensure ZOffset = 0.\n\n"
        "BAD EXAMPLES are included below (Generated__*.txt) — these contain formatting and logic errors. "
        "Avoid their mistakes, such as redundant nesting, missing 'end', or invalid Include quoting.\n\n"
        "Output ONLY valid GWL code — no commentary, no markdown, no validation statements."
        """
RULES FOR VARIABLE DECLARATION:
- Every variable used in a loop or expression must be declared first with 'local'. THIS IS A MUST! DO NOT assume that just because a variable is used in a loop or expression, it has been declared. You must declare every variable used in a loop or expression.
- For example:
    local $r = 0
    local $c = 0
    local $cx = 0
    local $cy = 0
    for $r = 0 to $rows - 1
        for $c = 0 to $cols - 1
            ...
        end
    end
"""

    )

    # === HUMAN MESSAGE ===
    human = f"""
USER DESIGN PROMPT:
{user_prompt}

GEOMETRY JSON:
{json.dumps(geom_data, indent=2)}

AVAILABLE META-ATOM FILES:
{include_snippet}

PATTERN MAP:
{pattern_text}

REFERENCE + BAD EXAMPLES (trimmed):
{docs_text}

FORMAT TARGET (use this as syntactic model):
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example Array of Cubes
% Objective: 63x
% Each cube: 4 µm
% Edge-to-edge spacing: 1 µm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
var $objective = 63
XOffset 0
YOffset 0
ZOffset 0
PowerScaling 0.6 ****Should always be this value
LaserPower 40
ScanSpeed 100
FindInterfaceAt 0.5
LineNumber 1
local $rows = 2
local $cols = 2
local $pitch = 5.0
local $xstart = -($pitch * ($cols - 1) / 2.0)
local $ystart = -($pitch * ($rows - 1) / 2.0)
for $r = 0 to $rows - 1
\tfor $c = 0 to $cols - 1
\t\tset $cx = $xstart + $c * $pitch
\t\tset $cy = $ystart + $r * $pitch
\t\tMoveStageX $cx
\t\tMoveStageY $cy
\t\tInclude meta_example.gwl
\t\tMoveStageX -$cx
\t\tMoveStageY -$cy
\tend
end
ZOffset 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Now produce the correct GWL array using all meta_*.gwl files according to the JSON pattern_map.
"""

    start_time = time.time()
    messages = [SystemMessage(content=system), HumanMessage(content=human)]
    result = llm.invoke(messages)
    elapsed = time.time() - start_time
    
    gwl_code = result.content.strip()

    # === Write Output ===
    fpath = project_dir / "main_array.gwl"
    fpath.write_text(gwl_code, encoding="utf-8")

    # Save metadata
    master_meta = {
        "filename": "main_array.gwl",
        "elapsed_time_sec": elapsed,
        "file_size_bytes": len(gwl_code),
        "array_config": {
            "rows": rows,
            "cols": cols,
            "spacing_um": spacing
        }
    }
    save_json_output(project_dir, "main_array.json", gwl_code, "master_gwl_builder", master_meta)

    print(f"\n Master GWL generated: {fpath}")
    print(f"Rows={rows}, Cols={cols}, Spacing={spacing:.2f} µm")
    
    state["master_gwl"] = str(fpath)
    state["master_gwl_metadata"] = master_meta
    return state


# ======================================================
# GRAPH DEFINITION
# ======================================================
builder = StateGraph(AgentState)
builder.add_node("geometry", node_geometry)
builder.add_node("intent", node_design_intent)
builder.add_node("meta_gwls", node_generate_meta_gwls)
builder.add_node("master_gwl", node_build_master_gwl)

builder.set_entry_point("geometry")
builder.add_edge("geometry", "intent")
builder.add_edge("intent", "meta_gwls")
builder.add_edge("meta_gwls", "master_gwl")
builder.add_edge("master_gwl", END)

compiled_graph = builder.compile()

# ======================================================
# RUN WRAPPER
# ======================================================
def run_agent(prompt_text: str):
    """
    Run the full Meta Array Design Agent and track total runtime and token usage.
    Writes geometry, meta-atom GWLs, and main array, while timing all stages.
    Generates PDF report and organizes all outputs by project.
    """

    import time
    import tiktoken

    # --- Begin timing ---
    start_time = time.time()

    # --- Initialize ---
    print("\n================ INITIALIZING META ARRAY DESIGN AGENT ================\n")
    input_prompt = prompt_text.strip()
    
    # Generate project name from prompt
    project_name = generate_project_name_from_prompt(input_prompt)
    project_dir = create_project_structure(project_name)
    
    print(f"Project: {project_name}")
    print(f"Output Directory: {project_dir}\n")
    
    # Save project overview
    project_overview = {
        "project_name": project_name,
        "created_at": datetime.now().isoformat(),
        "user_prompt": input_prompt
    }
    save_json_output(project_dir, "project_overview.json", project_overview, "initialization", {})

    # --- Run the pipeline graph ---
    result = compiled_graph.invoke({
        "user_prompt": input_prompt,
        "project_name": project_name,
        "project_dir": project_dir
    })

    # --- End timing ---
    total_time = time.time() - start_time

    # --- Accurate token counting ---
    # Collect metadata from each node to sum up actual tokens
    enc = tiktoken.get_encoding("cl100k_base")
    
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Add tokens from geometry node
    geom_meta = result.get("geometry_metadata", {})
    if geom_meta:
        total_output_tokens += len(enc.encode(result.get("geometry_json", "")))
    
    # Add tokens from design intent node  
    intent_meta = result.get("design_intent_metadata", {})
    if intent_meta:
        total_output_tokens += len(enc.encode(result.get("design_intent", "")))
    
    # Add tokens from meta GWL nodes
    meta_gwls_list = result.get("meta_gwls_metadata", [])
    for gwl_meta in meta_gwls_list:
        # Each meta-atom GWL file
        gwl_path = [p for p in result.get("meta_gwls", []) if f"meta_{gwl_meta['atom_id']}.gwl" in p]
        if gwl_path:
            gwl_content = Path(gwl_path[0]).read_text(encoding="utf-8")
            total_output_tokens += len(enc.encode(gwl_content))
    
    # Add tokens from master GWL node
    master_meta = result.get("master_gwl_metadata", {})
    if master_meta:
        master_path = result.get("master_gwl", "")
        if master_path and Path(master_path).exists():
            master_content = Path(master_path).read_text(encoding="utf-8")
            total_output_tokens += len(enc.encode(master_content))
    
    # Estimate input tokens (context sent to LLM each time)
    # Load docs to estimate their token contribution
    docs_text = load_docs_for_context()
    docs_tokens = len(enc.encode(docs_text))
    
    # Count number of LLM calls
    num_llm_calls = 0
    num_llm_calls += 1  # geometry node
    num_llm_calls += 1  # design intent node
    num_llm_calls += len(result.get("meta_gwls", []))  # one per meta-atom
    num_llm_calls += 1  # master GWL node
    
    # Each call includes: system prompt (~500 tokens) + user prompt (~50) + docs context
    avg_system_tokens = 500  # Approximate
    user_prompt_tokens = len(enc.encode(input_prompt))
    
    # Input tokens = (system + user + docs) × number of calls
    total_input_tokens = num_llm_calls * (avg_system_tokens + user_prompt_tokens + docs_tokens)
    
    total_tokens = total_input_tokens + total_output_tokens

    # --- Save metrics ---
    metrics = {
        "total_runtime_sec": round(total_time, 3),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "llm_calls": num_llm_calls,
        "docs_tokens_per_call": docs_tokens,
        "note": "Input tokens include full context (system prompt + user prompt + docs) for each LLM call"
    }


    save_json_output(project_dir, "run_stats.json", metrics, "final_metrics", {})

    # --- Generate PDF Report ---
    generate_pdf_report(project_dir, result, metrics)

    print("\n================ SUMMARY ================")
    print(f"Project: {project_name}")
    print(f"Location: {project_dir}")
    print(f"Total runtime: {metrics['total_runtime_sec']} s")
    print(f"Input tokens:  {metrics['input_tokens']}")
    print(f"Output tokens: {metrics['output_tokens']}")
    print(f"Total tokens:  {metrics['total_tokens']}")
    print("----------------------------------------")
    print("Generated Files:")
    print("  - project_report.pdf")
    print("  - project_overview.json")
    print("  - geometry_output.json")
    print("  - design_intent.json/.txt")
    print(f"  - {len(result.get('meta_gwls', []))} meta-atom GWL files")
    print("  - main_array.gwl")
    print("  - run_stats.json")
    print("========================================\n")

    return result

# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    # Read prompt from prompt.txt file
    prompt_file = SCRIPT_DIR / "prompt.txt"
    if prompt_file.exists():
        USER_PROMPT = prompt_file.read_text(encoding="utf-8").strip()
        # Remove quotes if present
        USER_PROMPT = USER_PROMPT.strip('"').strip("'")
    else:
        USER_PROMPT = "generate a 5x5 array of cylinders all of height 5, the radius should be 1 for each, the spacing center to center is 5"
    
    run_agent(USER_PROMPT)
