"""
Stage 1 (Part B): Control Flow Graph (CFG) Construction

This module builds Control Flow Graphs for functions using the AST.
Identifies:
- Basic blocks
- Guard conditions
- State mutations
- Error exits
- Control flow edges

NO LLM INFERENCE IN THIS STAGE.
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from agent5.clang_ast_parser import ASTNode, FunctionInfo, NodeType
from agent5.logging_utils import get_logger
from clang.cindex import CursorKind

logger = get_logger(__name__)


class EdgeType(Enum):
    """Types of control flow edges"""
    SEQUENTIAL = "sequential"  # Normal sequential flow
    TRUE_BRANCH = "true"  # True branch of condition
    FALSE_BRANCH = "false"  # False branch of condition
    LOOP_ENTRY = "loop_entry"  # Entry into loop
    LOOP_BACK = "loop_back"  # Back edge in loop
    LOOP_EXIT = "loop_exit"  # Exit from loop
    EXCEPTION = "exception"  # Exception handling edge
    RETURN = "return"  # Return from function


@dataclass
class BasicBlock:
    """Represents a basic block in the CFG"""
    id: int
    statements: List[ASTNode] = field(default_factory=list)
    predecessors: Set[int] = field(default_factory=set)
    successors: Set[int] = field(default_factory=set)
    edge_types: Dict[int, EdgeType] = field(default_factory=dict)  # successor_id -> edge_type
    
    # Semantic properties
    has_validation: bool = False
    has_state_mutation: bool = False
    has_side_effect: bool = False
    is_guard: bool = False
    is_error_exit: bool = False
    guard_condition: Optional[str] = None
    
    def add_successor(self, successor_id: int, edge_type: EdgeType):
        """Add a successor block"""
        self.successors.add(successor_id)
        self.edge_types[successor_id] = edge_type
        
    def add_predecessor(self, predecessor_id: int):
        """Add a predecessor block"""
        self.predecessors.add(predecessor_id)


@dataclass
class ControlFlowGraph:
    """Represents the Control Flow Graph of a function"""
    function_name: str
    blocks: Dict[int, BasicBlock] = field(default_factory=dict)
    entry_block_id: int = 0
    exit_block_ids: Set[int] = field(default_factory=set)
    
    def add_block(self, block: BasicBlock):
        """Add a basic block to the CFG"""
        self.blocks[block.id] = block
    
    def add_edge(self, from_id: int, to_id: int, edge_type: EdgeType):
        """Add an edge between two blocks"""
        if from_id in self.blocks and to_id in self.blocks:
            self.blocks[from_id].add_successor(to_id, edge_type)
            self.blocks[to_id].add_predecessor(from_id)
    
    def get_guard_blocks(self) -> List[BasicBlock]:
        """Get all blocks that contain guard conditions"""
        return [block for block in self.blocks.values() if block.is_guard]
    
    def get_state_mutation_blocks(self) -> List[BasicBlock]:
        """Get all blocks that contain state mutations"""
        return [block for block in self.blocks.values() if block.has_state_mutation]
    
    def get_error_exit_blocks(self) -> List[BasicBlock]:
        """Get all blocks that are error exits"""
        return [block for block in self.blocks.values() if block.is_error_exit]


class CFGBuilder:
    """
    Builds Control Flow Graphs from function ASTs.
    This is part of Stage 1: Full AST Construction.
    """
    
    def __init__(self):
        self.next_block_id = 0
    
    def build_cfg(self, func_info: FunctionInfo, ast_root: ASTNode) -> ControlFlowGraph:
        """
        Build a CFG for a function.
        
        Args:
            func_info: Information about the function
            ast_root: Root of the function's AST
            
        Returns:
            ControlFlowGraph for the function
        """
        logger.debug(f"Building CFG for function: {func_info.name}")
        
        cfg = ControlFlowGraph(function_name=func_info.qualified_name)
        self.next_block_id = 0
        
        # Create entry block
        entry_block = self._create_block()
        cfg.entry_block_id = entry_block.id
        cfg.add_block(entry_block)
        
        # Process function body
        if func_info.body_statements:
            self._process_statements(func_info.body_statements, entry_block, cfg)
        elif ast_root and ast_root.cursor:
            # Fallback: use ast_root's children if body_statements not available
            body_statements = ast_root.children
            if body_statements:
                self._process_statements(body_statements, entry_block, cfg)
        
        # Identify exit blocks (blocks with no successors or return statements)
        for block_id, block in cfg.blocks.items():
            if not block.successors:
                cfg.exit_block_ids.add(block_id)
            else:
                # Check if block has return statement
                for stmt in block.statements:
                    if stmt.node_type == NodeType.EARLY_EXIT:
                        cfg.exit_block_ids.add(block_id)
                        break
        
        # Analyze semantic properties of blocks
        self._analyze_block_semantics(cfg)
        
        logger.debug(f"CFG for {func_info.name}: {len(cfg.blocks)} blocks, "
                    f"{len(cfg.get_guard_blocks())} guards, "
                    f"{len(cfg.get_state_mutation_blocks())} state mutations")
        
        return cfg
    
    def _create_block(self) -> BasicBlock:
        """Create a new basic block with unique ID"""
        block = BasicBlock(id=self.next_block_id)
        self.next_block_id += 1
        return block
    
    def _process_statements(self, statements: List[ASTNode], current_block: BasicBlock, 
                           cfg: ControlFlowGraph) -> BasicBlock:
        """
        Process a list of statements and build CFG.
        
        Returns the last block in the sequence.
        """
        for stmt in statements:
            current_block = self._process_statement(stmt, current_block, cfg)
        
        return current_block
    
    def _process_statement(self, stmt: ASTNode, current_block: BasicBlock, 
                          cfg: ControlFlowGraph) -> BasicBlock:
        """
        Process a single statement and update CFG.
        
        Returns the block to continue with after this statement.
        """
        kind = stmt.kind
        
        # Decision points (if, switch)
        if kind == CursorKind.IF_STMT:
            return self._process_if_statement(stmt, current_block, cfg)
        
        elif kind == CursorKind.SWITCH_STMT:
            return self._process_switch_statement(stmt, current_block, cfg)
        
        # Loops
        elif kind in [CursorKind.FOR_STMT, CursorKind.WHILE_STMT, CursorKind.DO_STMT]:
            return self._process_loop_statement(stmt, current_block, cfg)
        
        # Early exits
        elif stmt.node_type == NodeType.EARLY_EXIT:
            current_block.statements.append(stmt)
            current_block.is_error_exit = self._is_error_exit(stmt)
            # No successors after early exit
            return current_block
        
        # Regular statements
        else:
            current_block.statements.append(stmt)
            return current_block
    
    def _process_if_statement(self, if_stmt: ASTNode, current_block: BasicBlock, 
                             cfg: ControlFlowGraph) -> BasicBlock:
        """Process an if statement and create CFG branches"""
        # Extract condition, then, and else parts
        children = if_stmt.children
        if len(children) < 2:
            # Malformed if statement
            current_block.statements.append(if_stmt)
            return current_block
        
        condition = children[0]
        then_body = children[1] if len(children) > 1 else None
        else_body = children[2] if len(children) > 2 else None
        
        # Add condition to current block
        current_block.statements.append(condition)
        current_block.is_guard = True
        current_block.guard_condition = condition.spelling
        
        # Create blocks for then and else branches
        then_block = self._create_block()
        cfg.add_block(then_block)
        cfg.add_edge(current_block.id, then_block.id, EdgeType.TRUE_BRANCH)
        
        # Process then branch
        if then_body:
            then_statements = [then_body] if then_body.kind != CursorKind.COMPOUND_STMT else then_body.children
            then_exit = self._process_statements(then_statements, then_block, cfg)
        else:
            then_exit = then_block
        
        # Process else branch if it exists
        if else_body:
            else_block = self._create_block()
            cfg.add_block(else_block)
            cfg.add_edge(current_block.id, else_block.id, EdgeType.FALSE_BRANCH)
            
            else_statements = [else_body] if else_body.kind != CursorKind.COMPOUND_STMT else else_body.children
            else_exit = self._process_statements(else_statements, else_block, cfg)
        else:
            # No else branch - false branch goes to merge block
            else_exit = self._create_block()
            cfg.add_block(else_exit)
            cfg.add_edge(current_block.id, else_exit.id, EdgeType.FALSE_BRANCH)
        
        # Create merge block
        merge_block = self._create_block()
        cfg.add_block(merge_block)
        
        # Connect branches to merge block
        if then_exit and then_exit.id not in cfg.exit_block_ids:
            cfg.add_edge(then_exit.id, merge_block.id, EdgeType.SEQUENTIAL)
        if else_exit and else_exit.id not in cfg.exit_block_ids:
            cfg.add_edge(else_exit.id, merge_block.id, EdgeType.SEQUENTIAL)
        
        return merge_block
    
    def _process_switch_statement(self, switch_stmt: ASTNode, current_block: BasicBlock, 
                                 cfg: ControlFlowGraph) -> BasicBlock:
        """Process a switch statement"""
        # Simplified: treat as if-else chain
        # In a full implementation, we'd handle case labels explicitly
        current_block.statements.append(switch_stmt)
        current_block.is_guard = True
        
        # Create merge block for after switch
        merge_block = self._create_block()
        cfg.add_block(merge_block)
        
        # Process cases (simplified)
        for child in switch_stmt.children:
            if child.kind in [CursorKind.CASE_STMT, CursorKind.DEFAULT_STMT]:
                case_block = self._create_block()
                cfg.add_block(case_block)
                cfg.add_edge(current_block.id, case_block.id, EdgeType.TRUE_BRANCH)
                
                case_statements = child.children
                case_exit = self._process_statements(case_statements, case_block, cfg)
                
                if case_exit and case_exit.id not in cfg.exit_block_ids:
                    cfg.add_edge(case_exit.id, merge_block.id, EdgeType.SEQUENTIAL)
        
        return merge_block
    
    def _process_loop_statement(self, loop_stmt: ASTNode, current_block: BasicBlock, 
                               cfg: ControlFlowGraph) -> BasicBlock:
        """Process a loop statement (for/while/do-while)"""
        # Create loop header block
        loop_header = self._create_block()
        cfg.add_block(loop_header)
        cfg.add_edge(current_block.id, loop_header.id, EdgeType.LOOP_ENTRY)
        
        # Add loop condition to header
        condition = None
        body = None
        
        if loop_stmt.kind == CursorKind.FOR_STMT:
            # For loop: init, condition, increment, body
            children = loop_stmt.children
            if len(children) >= 2:
                condition = children[1] if len(children) > 1 else None
                body = children[-1]
        elif loop_stmt.kind in [CursorKind.WHILE_STMT, CursorKind.DO_STMT]:
            # While/do-while: condition, body
            children = loop_stmt.children
            if len(children) >= 2:
                condition = children[0]
                body = children[1]
        
        if condition:
            loop_header.statements.append(condition)
            loop_header.is_guard = True
            loop_header.guard_condition = condition.spelling
        
        # Create loop body block
        body_block = self._create_block()
        cfg.add_block(body_block)
        cfg.add_edge(loop_header.id, body_block.id, EdgeType.TRUE_BRANCH)
        
        # Process loop body
        if body:
            body_statements = [body] if body.kind != CursorKind.COMPOUND_STMT else body.children
            body_exit = self._process_statements(body_statements, body_block, cfg)
            
            # Add back edge
            if body_exit and body_exit.id not in cfg.exit_block_ids:
                cfg.add_edge(body_exit.id, loop_header.id, EdgeType.LOOP_BACK)
        
        # Create loop exit block
        exit_block = self._create_block()
        cfg.add_block(exit_block)
        cfg.add_edge(loop_header.id, exit_block.id, EdgeType.FALSE_BRANCH)
        
        return exit_block
    
    def _is_error_exit(self, stmt: ASTNode) -> bool:
        """Determine if a statement is an error exit"""
        # Check for throw statements
        if stmt.kind == CursorKind.CXX_THROW_EXPR:
            return True
        
        # Check for return statements with error codes
        if stmt.kind == CursorKind.RETURN_STMT:
            # Look for negative returns, false, nullptr, error codes, etc.
            spelling = stmt.spelling.lower()
            if any(kw in spelling for kw in ['false', 'null', 'error', '-1', 'fail']):
                return True
        
        return False
    
    def _analyze_block_semantics(self, cfg: ControlFlowGraph):
        """Analyze semantic properties of each block"""
        for block in cfg.blocks.values():
            for stmt in block.statements:
                # Check for validation
                if stmt.node_type == NodeType.VALIDATION:
                    block.has_validation = True
                
                # Check for state mutation
                if stmt.node_type == NodeType.STATE_MUTATION:
                    block.has_state_mutation = True
                
                # Check for side effects
                if stmt.node_type == NodeType.IRREVERSIBLE_SIDE_EFFECT:
                    block.has_side_effect = True


def build_project_cfgs(call_graph) -> Dict[str, ControlFlowGraph]:
    """
    Build CFGs for all functions in a project.
    
    Args:
        call_graph: CallGraph from ClangASTParser
        
    Returns:
        Dictionary mapping function qualified names to their CFGs
    """
    from agent5.clang_ast_parser import ClangASTParser
    
    cfgs = {}
    builder = CFGBuilder()
    
    logger.info(f"Building CFGs for {len(call_graph.functions)} functions")
    
    for func_name, func_info in call_graph.functions.items():
        if func_info.ast_root:
            try:
                cfg = builder.build_cfg(func_info, func_info.ast_root)
                cfgs[func_name] = cfg
            except Exception as e:
                logger.error(f"Failed to build CFG for {func_name}: {e}")
    
    logger.info(f"Successfully built {len(cfgs)} CFGs")
    return cfgs


