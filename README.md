# Nanoscribe Design Agent - Setup & Usage Guide

## Prerequisites
- Python 3.8 or higher
- OpenAI API key

## Installation

### 1. Install Dependencies
Open PowerShell/Terminal in the `Agents_Design` folder and run:

```powershell
pip install -r requirements.txt
```

This will install:
- `langgraph` - Agent workflow framework
- `langchain-openai` - OpenAI integration
- `langchain-core` - Core LangChain functionality
- `reportlab` - PDF report generation
- `PyMuPDF` - PDF reading (for docs)
- `tiktoken` - Token counting

### 2. Set Up API Key

Place your OpenAI API key in:
```
Agentic Design/API.txt
```

The file should contain ONLY your API key (no quotes, no extra text):
```
sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Set Up Documentation (if you have Docs folder)

Place any reference documentation in:
```
Agentic Design/Docs/
```

The agent will automatically load `.txt` and `.pdf` files from this folder.

---

## Usage

### Method 1: Edit prompt.txt (Recommended)

1. **Edit your prompt** in `prompt.txt`:
   ```
   generate a 5x5 array of cylinders all of height 5, the radius should be 1 for each, the spacing center to center is 5
   ```

2. **Run the script**:
   ```powershell
   python FP_DESIGN_W_ERROR.py
   ```

### Method 2: Command Line

```powershell
python FP_DESIGN_W_ERROR.py
```

(It will read from `prompt.txt` automatically)

---

## Output

All outputs are saved to:
```
Agents_Design/Outputs/[project_name_timestamp]/
```

Generated files:
- `project_report.pdf` - Complete summary with visuals
- `project_overview.json` - Project metadata
- `geometry_output.json` - Planned geometry
- `design_intent.json/.txt` - Design summary
- `meta_*.gwl` - Individual meta-atom files
- `main_array.gwl` - Master array file
- `run_stats.json` - **Token usage & performance metrics**

---

## Troubleshooting

### Missing Dependencies
```powershell
pip install --upgrade -r requirements.txt
```

### API Key Error
Check that `../API.txt` exists (one folder up from `Agents_Design`)

### Path Issues
Ensure your folder structure is:
```
Agentic Design/
├── API.txt
├── Docs/
└── Agents_Design/
    ├── FP_DESIGN_W_ERROR.py
    ├── prompt.txt
    ├── requirements.txt
    └── Outputs/
```

---

## Example Prompts

Try these in `prompt.txt`:

```
generate a 3x3 array of cubes with side length 2um, spacing 10um center to center

create a 5x5 hexagonal array of cylinders, height 3um, radius 0.5um, pitch 8um

make a gradient array where column 1 has height 1um and column 5 has height 5um
```

---

## Monitoring Token Usage

After each run, check `Outputs/[project]/run_stats.json`:

```json
{
  "total_runtime_sec": 45.2,
  "input_tokens": 34200,
  "output_tokens": 8500,
  "total_tokens": 42700,
  "llm_calls": 4,
  "note": "Input tokens include full context..."
}
```

This now accurately reflects your OpenAI API usage!
