grammar GWL

// Entry point - a GWL file is a sequence of statements
entry Model: (stmts+=Stmt)*;

// Statements - order matters for parser disambiguation
Stmt: 
      VarDecl 
    | Assignment 
    | ForLoop 
    | IfBlock 
    | IncludeStmt 
    | CoordinateLine
    | CmdStmt
;

// Variable declaration: var $name = value OR local $name = value  
VarDecl: 
    kind=('var' | 'local') name=VarName '=' value=Expr
;

// Assignment: set $name = value
Assignment: 
    'set' name=VarName '=' value=Expr
;

// For loop: for $i = start to end [step stepVal] ... end
ForLoop: 
    'for' var=VarName '=' from=Expr 'to' to=Expr ('step' step=Expr)? 
    (body+=Stmt)* 
    'end'
;

// If block: if condition ... [else ...] end
IfBlock: 
    'if' cond=Condition 
    (thenBody+=Stmt)* 
    ('else' (elseBody+=Stmt)*)? 
    'end'
;

// Include statement: include path/to/file.gwl
IncludeStmt: 
    'include' file=FilePath
;

// Coordinate line: 3 or 4 expressions where first is NOT a bare ID
// Examples: $x 0 0, 10 12 0, sin(0) 1 2
CoordinateLine:
    x=CoordStartExpr y=Expr z=Expr (lp=Expr)?
;

// CoordStartExpr - expression that cannot start with a bare ID
CoordStartExpr infers Expr:
    CoordAdd
;

CoordAdd infers Expr:
    CoordMult ({infer BinExpr.left=current} op=('+' | '-') right=CoordMult)*
;

CoordMult infers Expr:
    CoordUnary ({infer BinExpr.left=current} op=('*' | '/') right=CoordUnary)*
;

CoordUnary infers Expr:
    NegExpr | CoordPrimary
;

// CoordPrimary - primary expressions that can START a coordinate line
// These must NOT match a bare ID (which would be a command)
// Removed FuncCall because ID '(' lookahead causes issues with backtracking
CoordPrimary infers Expr:
    Lit | VarRef | Group
;

// Command statement: CommandName [args...] [# (formatArgs)]
CmdStmt: 
    name=ID (args+=Expr)* ('#' '(' formatArgs+=Expr (',' formatArgs+=Expr)* ')')?
;

// Condition for if statements
Condition: 
    left=Expr op=CompareOp right=Expr
;

CompareOp returns string: 
    '==' | '!=' | '<=' | '>=' | '<' | '>'
;

// Full expression hierarchy
Expr: Add;

Add infers Expr:
    Mult ({infer BinExpr.left=current} op=('+' | '-') right=Mult)*
;

Mult infers Expr:
    Unary ({infer BinExpr.left=current} op=('*' | '/') right=Unary)*
;

Unary infers Expr:
    NegExpr | Primary
;

// Primary expressions - defined as separate rules
// Order matters: most specific first
Primary infers Expr:
    Lit | StringLit | VarRef | Group | FuncCall
;

// Literal number
Lit: val=NUMBER;

// String literal
StringLit: val=STRING;

// Variable reference: $name
VarRef: var=VarName;

// Grouped expression
Group: '(' expr=Expr ')';

// Negation expression
NegExpr: '-' operand=Primary;

// Function call: sin($x), mod($a, 2), etc.
// Must have ID followed immediately by '('
FuncCall: func=ID '(' (args+=Expr (',' args+=Expr)*)? ')';

// Variable name with $ prefix
VarName: '$' name=ID;

// File path for include
FilePath: 
    segments+=PathSegment (('/' | '\\') segments+=PathSegment)*
;

PathSegment: 
    name=ID ('.' ext=ID)?
;

// Terminal rules - order matters!
// Keywords are automatically handled by Langium when used in quotes
terminal NUMBER returns number: /[0-9]+(\.[0-9]+)?/;
terminal STRING: /"[^"]*"/;

// ID must come after NUMBER so numbers aren't matched as IDs
// ID explicitly excludes keywords by using a negative lookahead pattern
terminal ID returns string: /[_a-zA-Z][_a-zA-Z0-9]*/;

// Hidden terminals
hidden terminal WS: /\s+/;
hidden terminal SL_COMMENT: /%[^\n\r]*/;
