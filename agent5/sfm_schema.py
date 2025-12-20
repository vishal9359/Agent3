"""
Stage 4: Scenario Flow Model Construction (SINGLE SOURCE OF TRUTH)
Convert aggregated semantics into a deterministic Scenario Flow Model
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import json

from agent5.aggregation_engine import AggregatedSemantics, SemanticSummary


class StepType(Enum):
    """Types of steps in scenario flow"""
    START = "start"
    END = "end"
    DECISION = "decision"
    ACTION = "action"
    VALIDATION = "validation"
    STATE_CHANGE = "state_change"
    ERROR = "error"


class DetailLevel(Enum):
    """Detail levels for scenario flow"""
    HIGH = "high"  # Business-level steps only
    MEDIUM = "medium"  # + Decisions, validations, state changes
    DEEP = "deep"  # + Critical sub-operations affecting control/state


@dataclass
class ScenarioStep:
    """
    A single step in the Scenario Flow Model
    Deterministic representation - single source of truth
    """
    step_id: str
    step_type: StepType
    label: str  # Human-readable label
    description: str  # Detailed description
    detail_levels: List[DetailLevel]  # Which detail levels include this step
    on_success: Optional[str] = None  # Next step ID on success
    on_failure: Optional[str] = None  # Next step ID on failure
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioFlowModel:
    """
    Complete Scenario Flow Model for an entry point
    Single source of truth for diagram generation
    """
    scenario_name: str
    entry_function: str
    steps: Dict[str, ScenarioStep]  # step_id -> step
    start_step: str
    end_steps: List[str]
    
    def to_json(self) -> str:
        """Serialize to JSON"""
        data = {
            "scenario_name": self.scenario_name,
            "entry_function": self.entry_function,
            "start_step": self.start_step,
            "end_steps": self.end_steps,
            "steps": {
                step_id: {
                    "step_id": step.step_id,
                    "step_type": step.step_type.value,
                    "label": step.label,
                    "description": step.description,
                    "detail_levels": [dl.value for dl in step.detail_levels],
                    "on_success": step.on_success,
                    "on_failure": step.on_failure,
                    "metadata": step.metadata
                }
                for step_id, step in self.steps.items()
            }
        }
        return json.dumps(data, indent=2)
    
    @staticmethod
    def from_json(json_str: str) -> 'ScenarioFlowModel':
        """Deserialize from JSON"""
        data = json.loads(json_str)
        
        steps = {}
        for step_id, step_data in data["steps"].items():
            steps[step_id] = ScenarioStep(
                step_id=step_data["step_id"],
                step_type=StepType(step_data["step_type"]),
                label=step_data["label"],
                description=step_data["description"],
                detail_levels=[DetailLevel(dl) for dl in step_data["detail_levels"]],
                on_success=step_data.get("on_success"),
                on_failure=step_data.get("on_failure"),
                metadata=step_data.get("metadata", {})
            )
        
        return ScenarioFlowModel(
            scenario_name=data["scenario_name"],
            entry_function=data["entry_function"],
            steps=steps,
            start_step=data["start_step"],
            end_steps=data["end_steps"]
        )
    
    def validate(self) -> bool:
        """
        Validate SFM structure
        Ensures exactly one start, all paths end, etc.
        """
        # Must have exactly one start step
        start_steps = [s for s in self.steps.values() if s.step_type == StepType.START]
        if len(start_steps) != 1:
            print(f"ERROR: Expected exactly 1 start step, found {len(start_steps)}")
            return False
        
        # Must have at least one end step
        end_steps = [s for s in self.steps.values() if s.step_type == StepType.END]
        if len(end_steps) == 0:
            print(f"ERROR: No end steps found")
            return False
        
        # All steps must be reachable from start
        reachable = self._find_reachable_steps(self.start_step)
        unreachable = set(self.steps.keys()) - reachable
        if unreachable:
            print(f"WARNING: Unreachable steps: {unreachable}")
        
        # All non-end steps must have at least one successor
        for step_id, step in self.steps.items():
            if step.step_type != StepType.END:
                if not step.on_success and not step.on_failure:
                    print(f"ERROR: Non-end step {step_id} has no successors")
                    return False
        
        return True
    
    def _find_reachable_steps(self, start_step_id: str) -> set:
        """Find all steps reachable from start"""
        reachable = set()
        queue = [start_step_id]
        
        while queue:
            step_id = queue.pop(0)
            if step_id in reachable or step_id not in self.steps:
                continue
            
            reachable.add(step_id)
            step = self.steps[step_id]
            
            if step.on_success:
                queue.append(step.on_success)
            if step.on_failure:
                queue.append(step.on_failure)
        
        return reachable


class SFMBuilder:
    """
    Builds Scenario Flow Model from aggregated semantics
    NO LLM - purely deterministic conversion
    """
    
    def __init__(self):
        self.step_counter = 0
    
    def build_sfm(
        self,
        aggregated: AggregatedSemantics,
        scenario_name: Optional[str] = None
    ) -> ScenarioFlowModel:
        """
        Convert aggregated semantics into Scenario Flow Model
        One SFM per entry-point scenario
        """
        print("🔍 Stage 4: Scenario Flow Model Construction")
        
        if not scenario_name:
            scenario_name = f"Scenario: {aggregated.entry_function}"
        
        steps = {}
        
        # Create START step
        start_step_id = self._next_step_id()
        steps[start_step_id] = ScenarioStep(
            step_id=start_step_id,
            step_type=StepType.START,
            label=f"Start: {aggregated.entry_function}",
            description=aggregated.entry_summary.summary,
            detail_levels=[DetailLevel.HIGH, DetailLevel.MEDIUM, DetailLevel.DEEP]
        )
        
        # Convert entry summary to steps
        entry_steps = self._summary_to_steps(
            aggregated.entry_summary,
            aggregated.all_summaries
        )
        steps.update(entry_steps)
        
        # Link start to first entry step
        if entry_steps:
            first_step_id = next(iter(entry_steps.keys()))
            steps[start_step_id].on_success = first_step_id
        
        # Create END step
        end_step_id = self._next_step_id()
        steps[end_step_id] = ScenarioStep(
            step_id=end_step_id,
            step_type=StepType.END,
            label="End",
            description="Scenario completed successfully",
            detail_levels=[DetailLevel.HIGH, DetailLevel.MEDIUM, DetailLevel.DEEP]
        )
        
        # Link last steps to end
        for step in entry_steps.values():
            if step.on_success is None and step.step_type != StepType.END:
                step.on_success = end_step_id
        
        # Create error end step
        error_step_id = self._next_step_id()
        steps[error_step_id] = ScenarioStep(
            step_id=error_step_id,
            step_type=StepType.END,
            label="Error",
            description="Scenario failed",
            detail_levels=[DetailLevel.HIGH, DetailLevel.MEDIUM, DetailLevel.DEEP],
            metadata={"is_error": True}
        )
        
        # Link error conditions to error end
        for step in entry_steps.values():
            if step.on_failure is None and step.step_type in {StepType.DECISION, StepType.VALIDATION}:
                step.on_failure = error_step_id
        
        sfm = ScenarioFlowModel(
            scenario_name=scenario_name,
            entry_function=aggregated.entry_function,
            steps=steps,
            start_step=start_step_id,
            end_steps=[end_step_id, error_step_id]
        )
        
        # Validate
        if sfm.validate():
            print(f"✅ Stage 4 complete: Built SFM with {len(steps)} steps")
        else:
            print(f"⚠️ Stage 4 warning: SFM validation failed")
        
        return sfm
    
    def _summary_to_steps(
        self,
        summary: SemanticSummary,
        all_summaries: Dict[str, SemanticSummary]
    ) -> Dict[str, ScenarioStep]:
        """Convert semantic summary to scenario steps"""
        steps = {}
        prev_step_id = None
        
        # Preconditions -> Validation steps (medium, deep)
        for precond in summary.preconditions:
            step_id = self._next_step_id()
            steps[step_id] = ScenarioStep(
                step_id=step_id,
                step_type=StepType.VALIDATION,
                label=f"Validate: {self._shorten(precond)}",
                description=precond,
                detail_levels=[DetailLevel.MEDIUM, DetailLevel.DEEP]
            )
            
            if prev_step_id and prev_step_id in steps:
                steps[prev_step_id].on_success = step_id
            
            prev_step_id = step_id
        
        # Control flow -> Decision steps (high for major, medium for all)
        for ctrl_flow in summary.control_flow:
            is_major = self._is_major_decision(ctrl_flow)
            
            step_id = self._next_step_id()
            steps[step_id] = ScenarioStep(
                step_id=step_id,
                step_type=StepType.DECISION,
                label=f"Decision: {self._shorten(ctrl_flow)}",
                description=ctrl_flow,
                detail_levels=[DetailLevel.HIGH] if is_major else [DetailLevel.MEDIUM, DetailLevel.DEEP],
                metadata={"is_major": is_major}
            )
            
            if prev_step_id and prev_step_id in steps:
                steps[prev_step_id].on_success = step_id
            
            prev_step_id = step_id
        
        # Side effects -> State change steps (medium, deep)
        for side_effect in summary.side_effects:
            is_critical = self._is_critical_side_effect(side_effect)
            
            step_id = self._next_step_id()
            steps[step_id] = ScenarioStep(
                step_id=step_id,
                step_type=StepType.STATE_CHANGE,
                label=f"Change: {self._shorten(side_effect)}",
                description=side_effect,
                detail_levels=[DetailLevel.DEEP] if is_critical else [DetailLevel.MEDIUM, DetailLevel.DEEP],
                metadata={"is_critical": is_critical}
            )
            
            if prev_step_id and prev_step_id in steps:
                steps[prev_step_id].on_success = step_id
            
            prev_step_id = step_id
        
        # Postconditions -> Action steps (high)
        if summary.postconditions:
            step_id = self._next_step_id()
            postcond_summary = "; ".join(summary.postconditions[:2])  # First 2
            steps[step_id] = ScenarioStep(
                step_id=step_id,
                step_type=StepType.ACTION,
                label=f"Result: {self._shorten(postcond_summary)}",
                description=postcond_summary,
                detail_levels=[DetailLevel.HIGH, DetailLevel.MEDIUM, DetailLevel.DEEP]
            )
            
            if prev_step_id and prev_step_id in steps:
                steps[prev_step_id].on_success = step_id
            
            prev_step_id = step_id
        
        return steps
    
    def _is_major_decision(self, control_flow: str) -> bool:
        """Determine if a control flow is a major business decision"""
        major_keywords = ['business', 'strategy', 'policy', 'authorize', 'approve', 'reject']
        return any(keyword in control_flow.lower() for keyword in major_keywords)
    
    def _is_critical_side_effect(self, side_effect: str) -> bool:
        """Determine if a side effect is critical"""
        critical_keywords = ['irreversible', 'delete', 'commit', 'finalize', 'publish', 'execute']
        return any(keyword in side_effect.lower() for keyword in critical_keywords)
    
    def _shorten(self, text: str, max_len: int = 50) -> str:
        """Shorten text for labels"""
        if len(text) <= max_len:
            return text
        return text[:max_len-3] + "..."
    
    def _next_step_id(self) -> str:
        """Generate next step ID"""
        self.step_counter += 1
        return f"S{self.step_counter}"





