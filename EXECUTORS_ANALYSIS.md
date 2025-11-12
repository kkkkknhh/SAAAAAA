# Comprehensive Analysis: Executors.py Architecture & Factory Alignment

## Executive Summary

The executors system is a sophisticated multi-layered orchestration framework for executing policy analysis tasks with 30 specialized executors (D1Q1-D6Q5). The system attempts to integrate frontier paradigms (quantum optimization, neuromorphic computing, causal inference) with practical policy document analysis, but has **significant architecture misalignment issues** between factory and executor contracts.

---

## 1. KEY CLASSES AND THEIR RESPONSIBILITIES

### 1.1 Base Infrastructure

#### **ExecutorBase (Abstract Base Class)**
- **Responsibility**: Define validation contracts for all executors
- **Key Methods**:
  - `validate_before_execution()` - Pre-flight checks (dependencies, calibrations, resources)
  - `_check_dependencies()` - Verify class registry contains required classes
  - `_check_calibration()` - Verify all methods have explicit calibrations
  - `_check_resources()` - Verify config, signal registry, method executor available
  - `_get_method_sequence()` - Abstract method, must be overridden by subclasses

**Contract Enforcement**: Returns `ValidationResult` dataclass with:
- `is_valid: bool` - Validation passed/failed
- `severity: str` - ERROR, WARNING, INFO
- `message: str` - Detailed message
- `context: dict` - Structured context data

#### **MethodSequenceValidatingMixin**
- **Responsibility**: Validate method sequences before execution
- **Key Methods**:
  - `_validate_method_sequences()` - Ensure all (class, method) pairs are callable
  - `_get_method_sequence()` - Supports both CLASS ATTRIBUTE and METHOD IMPLEMENTATION patterns
- **Dual Support Pattern** (Lines 1307-1317):
  ```python
  # NEW: Check for METHOD_SEQUENCE class attribute first
  if hasattr(self.__class__, 'METHOD_SEQUENCE'):
      return self.__class__.METHOD_SEQUENCE
  # Fallback to method implementation
  return []
  ```

### 1.2 Advanced Executor Implementation

#### **AdvancedDataFlowExecutor(ExecutorBase, MethodSequenceValidatingMixin)**
- **Responsibility**: Core execution engine with frontier paradigm integration
- **Key Attributes**:
  - `config: ExecutorConfig` - REQUIRED (raises ValueError if None)
  - `executor: MethodExecutor` - Method registry and execution interface
  - `signal_registry` - Optional signal infrastructure
  - `calibration: CalibrationOrchestrator` - NEW: Calibration integration
  
- **Key Methods**:
  - `execute_with_optimization()` - Main execution pipeline with:
    - Calibration phase (skip methods with score < 0.3)
    - Signal fetching and consumption tracking
    - Deterministic seeding (policy_unit_id + correlation_id)
    - Retry logic (max 3 attempts per method)
    - Advanced module activation tracking
  - `_validate_executor_config()` - Enforce config constraints at construction
  - `_validate_calibrations()` - Strict validation of calibration entries
  - `_fetch_signals()` - Signal retrieval with OpenTelemetry spans
  - `_prepare_arguments()` - Argument context building
  - `_assess_data_quality()` - Quality scoring for neuromorphic processing

- **Constructor Signature** (Line 1326):
  ```python
  def __init__(
      self,
      method_executor,                    # REQUIRED
      signal_registry=None,               # Optional
      config: ExecutorConfig | None = None,  # REQUIRED (enforced)
      questionnaire_provider=None,        # Optional
      calibration_orchestrator: CalibrationOrchestrator | None = None,  # NEW
  )
  ```

### 1.3 Specialized Executors (30 Subclasses)

#### Pattern: D1Q1_Executor through D6Q5_Executor
- **Responsibility**: Execute specific question-dimension combinations
- **Structure**:
  ```python
  class D1Q1_Executor(AdvancedDataFlowExecutor):
      def __init__(self, method_executor, signal_registry=None, 
                   config: ExecutorConfig | None = None, 
                   calibration_orchestrator: CalibrationOrchestrator | None = None):
          super().__init__(method_executor, signal_registry, config, calibration_orchestrator)
          self._validate_method_sequences()    # Validation at construction
          self._validate_calibrations()
      
      def _get_method_sequence(self) -> list[tuple[str, str]]:
          return [('Class1', 'method1'), ('Class2', 'method2'), ...]
      
      def execute(self, doc, method_executor):
          method_sequence = self._get_method_sequence()
          return self.execute_with_optimization(doc, method_executor, method_sequence)
  ```

- **Two Patterns Observed**:
  1. **D1Q1 Pattern** (with validation):
     - Implements `_get_method_sequence()` as class method
     - Calls validation in `__init__`
     - Calls `_get_method_sequence()` in `execute()`
  
  2. **D1Q2-D6Q5 Pattern** (inline sequences):
     - Defines method_sequence inline in `execute()` method
     - No `_get_method_sequence()` implementation
     - Less validation-friendly

### 1.4 Advanced Paradigm Modules

#### **QuantumExecutionOptimizer**
- Uses Grover-inspired quantum search for method path selection
- Tracks execution history and applies oracle + diffusion operators
- **Activation**: Triggered when num_methods >= 3

#### **NeuromorphicFlowController**
- Spiking neurons with spike-timing-dependent plasticity (STDP)
- Data quality assessment and adaptive flow processing
- **Activation**: On every data flow processing

#### **CausalGraph**
- PC algorithm for causal structure learning
- Partial correlation independence testing
- Topological execution order determination
- **Activation**: When optimizing execution order for 2+ questions

#### **InformationFlowOptimizer**
- Shannon entropy calculation for data flow
- Mutual information matrix tracking
- Information bottleneck detection
- **Activation**: Continuous entropy tracking during execution

#### **MetaLearningStrategy**
- Epsilon-greedy strategy selection with exponential moving average
- 5 configurable execution strategies (parallel/batch/pruning combinations)
- **Activation**: On every execution for adaptive strategy selection

#### **AttentionMechanism**
- Query-key-value projection for method prioritization
- Multi-head attention simulation
- **Activation**: For focusing computational resources on critical methods

#### **CategoryTheoryExecutor** & **ProbabilisticExecutor**
- Theoretical constructs for composable execution pipelines
- Probabilistic programming with Bayesian updates
- No direct parameterization (theoretical only)

---

## 2. FACTORY INTERACTION PATTERNS

### 2.1 Factory Architecture (factory.py)

#### **ProcessorBundle (Data Carrier)**
```python
@dataclass(frozen=True)
class ProcessorBundle:
    method_executor: MethodExecutor
    questionnaire: Mapping[str, Any]
    factory: CoreModuleFactory
```

#### **build_processor() Function** (Lines 630-698)
```python
def build_processor(
    *, 
    questionnaire_path: Path | None = None,
    data_dir: Path | None = None,
    factory: Optional[CoreModuleFactory] = None,
    enable_signals: bool = True,
) -> ProcessorBundle
```

**Steps**:
1. Create or use provided CoreModuleFactory
2. Load questionnaire (cached with global provider update)
3. Create immutable questionnaire snapshot (MappingProxyType)
4. Build signal infrastructure (if enabled)
5. Create MethodExecutor with signal_registry
6. Return ProcessorBundle

**KEY ISSUE**: Factory does NOT construct executors - it only provides:
- Questionnaire data
- MethodExecutor instance
- CoreModuleFactory reference

### 2.2 Executor Construction Gap

**Current Pattern**:
```python
# Factory creates the bundle
bundle = build_processor()

# Executor created SEPARATELY (outside factory)
executor = D1Q1_Executor(
    method_executor=bundle.method_executor,
    signal_registry=signal_registry,
    config=config,
    calibration_orchestrator=calibration
)
```

**Problem**: Factory and executors have DECOUPLED construction:
- Factory builds MethodExecutor with limited dependency knowledge
- Executors independently validate their full dependency chains
- No validation coupling between factory and executor contracts

### 2.3 MethodExecutor (core.py, Lines 810-964)

#### **Constructor** (Lines 817-901):
1. Builds class registry (`build_class_registry()`)
2. Instantiates ALL classes (with special handling for MunicipalOntology, PolicyTextProcessor)
3. Validates calibrations (NOW DEPRECATED - YAML loading disabled, lines 837-848)
4. Creates ExtendedArgRouter for method routing
5. Tracks degradation status and reasons

#### **Key Properties**:
- `instances: dict[str, Any]` - All instantiated classes
- `calibrations: dict[str, dict]` - Empty dict (YAML deprecated)
- `degraded_mode: bool` - Tracks instantiation failures
- `degraded_reasons: list[str]` - Reasons for degradation

#### **execute() Method** (Lines 922-964):
1. Fail-fast calibration enforcement (raises RuntimeError if missing/placeholder)
2. Get instance from registry
3. Route arguments via ExtendedArgRouter
4. Execute method with routed kwargs

**CRITICAL CONTRACT**: Callers MUST provide kwargs matching method signature

---

## 3. CONTRACTS AND INTERFACES

### 3.1 ExecutorConfig Contract

**Location**: `executor_config.py`

```python
class ExecutorConfig(BaseModel):
    max_tokens: int = 2048  # Range: 256-8192
    temperature: float = 0.0  # Range: 0.0-2.0
    timeout_s: float = 30.0  # Range: 1.0-300.0
    retry: int = 2  # Range: 0-5
    policy_area: PolicyArea | None = None
    regex_pack: list[str] = []
    thresholds: dict[str, float] = {}  # Range: [0.0, 1.0]
    entities_whitelist: list[str] = []
    enable_symbolic_sparse: bool = True
    seed: int = 0  # Range: 0-2^31-1
    advanced_modules: AdvancedModuleConfig | None = None
```

**Methods**:
- `from_env()` - Load from environment variables (EXECUTOR_* prefix)
- `from_cli_args()` - Load from CLI arguments
- `from_cli()` - Register with Typer app
- `describe()` - Human-readable configuration surface
- `merge_overrides()` - Deterministic config merging
- `compute_hash()` - BLAKE3 hash for fingerprinting
- `validate_latency_budget()` - Enforce retry * timeout_s < max_latency_s

**Key Property**: FROZEN model (immutable after construction)

### 3.2 Input/Output Contracts (TypedDict)

**Location**: `contracts.py` / `utils/core_contracts.py`

```python
# Input Contracts
CDAFFrameworkInputContract
ContradictionDetectorInputContract
DocumentData
EmbeddingPolicyInputContract
PDETAnalyzerInputContract
PolicyProcessorInputContract
SemanticAnalyzerInputContract
SemanticChunkingInputContract
TeoriaCambioInputContract

# Output Contracts (corresponding *OutputContract classes)
```

**Usage Pattern**:
- Factory constructs input contracts from documents
- Executors pass contracts to methods
- MethodExecutor routes kwargs based on contracts
- No explicit type checking at runtime (TypedDict is structural)

### 3.3 Calibration Contract

**Location**: `calibration_registry.py`

**Interface**:
```python
class CalibrationResult:
    final_score: float
    is_default_like() -> bool

def resolve_calibration(
    class_name: str,
    method_name: str,
    strict: bool = True
) -> CalibrationResult | None
```

**Contract Enforcement**:
- Executors MUST validate all methods have explicit calibrations
- Methods with `is_default_like() == True` raise RuntimeError
- MethodExecutor.execute() fails-fast on missing/placeholder calibrations
- CalibrationOrchestrator provides dynamic calibration during execution

### 3.4 Signal Consumption Contract

**Location**: `signal_consumption.py`

```python
class SignalConsumptionProof:
    executor_id: str
    question_id: str
    policy_area: str
    consumed_patterns: dict[str, int]
    
    def record_pattern_match(self, pattern: str, match: str) -> None
```

**Usage Pattern** (Lines 1852-1889):
1. Create SignalConsumptionProof for tracking
2. Fetch signals from registry by policy_area
3. Apply regex patterns to document
4. Record pattern matches in proof
5. Store in argument context for methods to access

---

## 4. ERROR-PRONE PATTERNS AND VULNERABILITIES

### 4.1 Critical Contract Violations

#### **Issue 1: Config Enforcement Violation**
**Location**: Line 1335-1340 (AdvancedDataFlowExecutor.__init__)

```python
if config is None:
    raise ValueError(f"...ExecutorConfig is required and cannot be None...")
# BUT THEN:
self.config = config or CONSERVATIVE_CONFIG  # Line 1345
```

**Problem**: 
- Code raises if config is None
- Then assigns `config or CONSERVATIVE_CONFIG`
- This is unreachable dead code - if config was None, ValueError already raised
- Line 1345 should be: `self.config = config` (no fallback needed)

**Impact**: Config None-safety appears enforced but the code is confusing

---

#### **Issue 2: Method Sequence Pattern Mismatch**
**Location**: Inconsistent across executor subclasses

**Pattern A (D1Q1)**: Implements `_get_method_sequence()` + validation
```python
def __init__(self, ...):
    super().__init__(...)
    self._validate_method_sequences()  # VALIDATION CALLED
    self._validate_calibrations()

def _get_method_sequence(self) -> list[tuple[str, str]]:
    return [...]  # Explicit method

def execute(self, doc, method_executor):
    method_sequence = self._get_method_sequence()
    return self.execute_with_optimization(...)
```

**Pattern B (D1Q2-D6Q5)**: Inline sequences, no early validation
```python
def __init__(self, ...):
    super().__init__(...)
    self._validate_calibrations()  # NO SEQUENCE VALIDATION

def execute(self, doc, method_executor):
    method_sequence = [(...), (...), ...]  # INLINE - not validated!
    return self.execute_with_optimization(...)
```

**Problem**:
- Pattern B executors skip `_validate_method_sequences()`
- Method sequence is defined inline in `execute()`, not validated at construction
- Errors only surface at execution time, not at instantiation time
- 29/30 executors use this pattern!

**Impact**: Late failure detection, harder debugging

---

#### **Issue 3: Validation Timing Inconsistency**
**Location**: Lines 1482-1484

```python
# NOTE: Validation NOT called in base class because most executors
# define method_sequence in execute(), not in _get_method_sequence().
# Executors that want validation must call it explicitly in their __init__.
```

**Acknowledgment**: Code even documents this issue!

**Problem**:
- Base class doesn't validate (can't, methods defined at runtime)
- Subclasses must remember to validate
- D1Q1 does, D1Q2-D6Q5 don't
- Silent contract violation

---

### 4.2 Factory-Executor Contract Misalignment

#### **Issue 4: Missing ExecutorConfig from Factory**
**Location**: `build_processor()` (factory.py)

**Problem**: 
```python
def build_processor(...) -> ProcessorBundle:
    # Creates MethodExecutor
    executor = MethodExecutor(signal_registry=signal_registry)
    return ProcessorBundle(method_executor=executor, ...)
```

**What's Missing**:
- Factory does NOT construct ExecutorConfig
- Factory does NOT pass config to executors
- Caller must separately create config and pass to executor:
  ```python
  bundle = build_processor()
  config = ExecutorConfig(timeout_s=30.0, ...)  # SEPARATE
  executor = D1Q1_Executor(bundle.method_executor, config=config)
  ```

**Contract Violation**: Factory + Executor have split responsibilities
- Factory: questionnaire, method_executor, signals
- Executor: config, validation, optimization
- No integrated factory for complete executor construction

---

#### **Issue 5: CalibrationOrchestrator Optional but Validation Mandatory**
**Location**: Lines 1348, 1722-1796

```python
# Constructor accepts optional calibration
self.calibration = calibration_orchestrator  # Can be None

# But validation requires it indirectly
self._validate_calibrations()  # Checks resolve_calibration()
```

**Problem**:
- Executor validates calibrations exist in global registry
- But CalibrationOrchestrator (optional) provides dynamic calibration
- If calibration orchestrator is None, execution skips calibration phase (lines 1794-1795)
- Methods may be skipped during execution if calibration scores too low
- Two different calibration sources not reconciled

**Contract Gap**: No explicit contract for calibration availability

---

### 4.3 Advanced Modules Validation Issues

#### **Issue 6: Undocumented Module Activation Conditions**
**Location**: Lines 13-22, module docstring

Documented conditions:
1. Quantum Optimization: num_methods >= 3 (line 15) ✓
2. Neuromorphic Computing: every data flow (line 16) ✓
3. Causal Inference: 2+ questions (line 17) ✓
4. Meta-Learning: every execution (line 18) ✓
5-9: Information Theory, Attention, Topological, Category, Probabilistic

**Problem**: 
- Activation conditions described in docstring
- No code enforcement of conditions
- Example: Quantum optimizer always initialized (line 1424), even for single method
- Neuromorphic always initialized (line 1431), even if not needed
- No runtime checks to activate/deactivate based on conditions

**Impact**: Wasted resources, misleading performance metrics

---

#### **Issue 7: Advanced Config Type Safety**
**Location**: Line 1357-1359

```python
adv_config: AdvancedModuleConfig = (
    self.config.advanced_modules or CONSERVATIVE_ADVANCED_CONFIG
)
```

**Problem**:
- Config field is `advanced_modules: AdvancedModuleConfig | None`
- Executor trusts Pydantic validation
- But no runtime assertions that advanced_modules is set
- If set to None and CONSERVATIVE_ADVANCED_CONFIG fails to load, silent failure

**Missing**: Type guard or assertion

---

### 4.4 Argument Preparation Vulnerabilities

#### **Issue 8: Argument Context Management**
**Location**: Lines 1476, 1903-1906

```python
self._argument_context: dict[str, Any] = {}

# Reset (line 1903)
self._reset_argument_context(doc)

# Re-add signals (lines 1904-1906)
if signals:
    self._argument_context['signals'] = signals
```

**Problem**:
- `_argument_context` is mutable dict
- Reset + re-add pattern is error-prone
- Signals added after reset but before methods execute
- If method sequence references stale context, will fail
- No atomic transaction - state inconsistency possible

**Missing**: Immutable context snapshots, atomic context updates

---

#### **Issue 9: Signal Fetching Exception Swallowing**
**Location**: Lines 1534-1625

```python
if self.signal_registry is None:
    logger.warning("Signal registry is explicitly None...")
    return None

# Try to fetch signals
signal_pack = self.signal_registry.get(policy_area)

# If not found:
logger.warning(f"Signal pack not found for policy_area='{policy_area}'...")
return None
```

**Problem**:
- Explicit None vs missing - same handling
- AttributeError from bad registry object converts to None return
- Methods expecting signals get None instead of error
- Silent degradation possible

**Missing**: Explicit error handling for registry contract violations

---

### 4.5 Deterministic Seeding Issues

#### **Issue 10: Deterministic Context Not Thread-Safe**
**Location**: Lines 118-159

```python
@contextmanager
def deterministic(policy_unit_id: str | None, correlation_id: str | None):
    # ...
    seeds = DeterministicSeeds(np=base_seed, python=base_seed + 1)
    random.seed(seeds.python)  # GLOBAL STATE MODIFICATION
    yield seeds
```

**Problem**:
- Sets global `random.seed()` (affects global random state)
- Not thread-safe for concurrent executions
- If two coroutines use deterministic() simultaneously, seeds may conflict

**Missing**: Thread-local or async-safe RNG setup

---

### 4.6 Retry Logic Defects

#### **Issue 11: Uninitialized Variable in Exception Handler**
**Location**: Lines 1952-2020 (approx)

```python
success = False
max_retries = 3
prepared_kwargs = {}  # INITIALIZED TO PREVENT UnboundLocalError

for attempt in range(max_retries):
    try:
        prepared_kwargs = self._prepare_arguments(...)
        result = self.executor.execute(...)
        # ...
        break
    except Exception as e:
        if attempt < max_retries - 1:
            # Retry
            pass
        else:
            # Final attempt failed
            logger.error(f"Method {method_key} failed after {max_retries} attempts",
                        extra={"prepared_kwargs": prepared_kwargs})  # MIGHT BE STALE
```

**Problem**:
- `prepared_kwargs` initialized to prevent UnboundLocalError (line 1953 comment)
- But if exception occurs during preparation, logged `prepared_kwargs` is empty dict
- Error logging loses actual failing arguments
- Makes debugging harder

**Missing**: Try-except around argument preparation to capture actual state

---

### 4.7 Metadata Tracking Issues

#### **Issue 12: Signal Usage Not Validated**
**Location**: Lines 1557-1566

```python
self.used_signals.append({
    "version": signal_pack.version,
    "policy_area": signal_pack.policy_area,
    "hash": signal_pack.compute_hash(),
    # ...
    "pattern_count": len(signal_pack.patterns) if hasattr(signal_pack, 'patterns') else 0,
})
```

**Problem**:
- Uses `hasattr()` to check for optional attributes
- If attribute exists but is wrong type, will fail silently or during len()
- `compute_hash()` might not exist on signal_pack
- No schema validation for signal_pack structure

**Missing**: SignalPack protocol/interface definition

---

### 4.8 Circular Import and Optional Dependency Risks

#### **Issue 13: Optional Module Dependencies**
**Location**: Lines 74-98

```python
try:
    from opentelemetry import trace
    HAS_OTEL = True
except ImportError:
    tracer = None
    HAS_OTEL = False

try:
    import networkx as nx
except Exception:
    nx = None

try:
    from saaaaaa.analysis.teoria_cambio import CategoriaCausal
except Exception:
    CategoriaCausal = None
```

**Problem**:
- Missing modules silently set to None
- Code checks `HAS_OTEL` before using tracer
- But some module-uses not gated (e.g., nx in CausalGraph)
- Partial degradation possible without clear error messages

**Missing**: Explicit dependency checking at executor init time

---

---

## 5. INNOVATIVE IMPROVEMENTS FOR FACTORY-EXECUTOR ALIGNMENT

### 5.1 Unified ExecutorBuilder Factory

**Current**: Decoupled - factory creates MethodExecutor, executor created separately

**Proposal**:
```python
class ExecutorBuilder:
    """Unified factory for executor construction with full contract validation."""
    
    def __init__(self, data_dir: Path | None = None):
        self.bundle = build_processor(data_dir=data_dir)
        self.config: ExecutorConfig | None = None
        self.calibration: CalibrationOrchestrator | None = None
    
    def with_config(self, config: ExecutorConfig) -> 'ExecutorBuilder':
        """Set executor config and validate."""
        config.validate_latency_budget()
        self.config = config
        return self
    
    def with_calibration(self, calib: CalibrationOrchestrator) -> 'ExecutorBuilder':
        """Attach calibration orchestrator."""
        self.calibration = calib
        return self
    
    def build_executor(self, executor_class: type[T]) -> T:
        """Construct executor with full contract validation."""
        if self.config is None:
            raise ValueError("Config required - call with_config() first")
        
        executor = executor_class(
            method_executor=self.bundle.method_executor,
            signal_registry=...,
            config=self.config,
            calibration_orchestrator=self.calibration
        )
        
        # Validate contracts
        executor.validate_before_execution()
        
        return executor

# Usage:
builder = ExecutorBuilder(data_dir=Path("./data"))
config = ExecutorConfig(timeout_s=30.0, seed=42)

executor = (builder
    .with_config(config)
    .with_calibration(calibration_orch)
    .build_executor(D1Q1_Executor))
```

---

### 5.2 Standardize Method Sequence Definition

**Current**: Two patterns (class method vs inline)

**Proposal**: Enforce `METHOD_SEQUENCE` class attribute
```python
class D1Q1_Executor(AdvancedDataFlowExecutor):
    # NEW: Explicit class attribute (validated at class definition time)
    METHOD_SEQUENCE: ClassVar[list[tuple[str, str]]] = [
        ('IndustrialPolicyProcessor', 'process'),
        ('PolicyTextProcessor', 'segment_into_sentences'),
        ...
    ]
    
    def __init__(self, ...):
        super().__init__(...)
        # Validation happens in base class now
        self._validate_method_sequences()  # Always safe
    
    def execute(self, doc, method_executor):
        # Use class attribute directly
        return self.execute_with_optimization(
            doc, method_executor, self.METHOD_SEQUENCE
        )

# Base class validates at init:
class AdvancedDataFlowExecutor(...):
    def __init__(self, ...):
        # Check CLASS ATTRIBUTE exists
        if not hasattr(self.__class__, 'METHOD_SEQUENCE'):
            raise TypeError(f"{self.__class__.__name__} missing METHOD_SEQUENCE")
        
        self._validate_method_sequences()  # Works on CLASS attribute
```

**Benefits**:
- Validation at construction time, not execution time
- Type hints for IDE autocompletion
- Static analysis friendly
- No runtime method lookups

---

### 5.3 Contract-First Executor Validation

**Current**: Multiple separate validation methods

**Proposal**: Single comprehensive `ExecutorContract` validation
```python
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class ExecutorContract:
    """Formal executor interface contract."""
    
    # Required resources
    method_executor: MethodExecutor  # Must have instances registry
    config: ExecutorConfig  # Must be valid and frozen
    signal_registry: SignalRegistry | None
    
    # Required behaviors
    method_sequence: list[tuple[str, str]]
    
    def validate(self) -> ValidationResult:
        """Comprehensive contract validation."""
        errors = []
        
        # Structural checks
        if not isinstance(self.config, ExecutorConfig):
            errors.append("config must be ExecutorConfig instance")
        
        if self.config.timeout_s <= 0:
            errors.append("timeout_s must be > 0")
        
        if self.config.seed < 0:
            errors.append("seed must be >= 0")
        
        # Dependency checks
        for class_name, method_name in self.method_sequence:
            if class_name not in self.method_executor.instances:
                errors.append(f"Missing class: {class_name}")
                continue
            
            instance = self.method_executor.instances[class_name]
            if not hasattr(instance, method_name):
                errors.append(f"{class_name} has no method {method_name}")
                continue
            
            if not callable(getattr(instance, method_name)):
                errors.append(f"{class_name}.{method_name} is not callable")
        
        # Calibration checks
        for class_name, method_name in self.method_sequence:
            calib = resolve_calibration(class_name, method_name, strict=False)
            if calib is None:
                errors.append(f"Missing calibration: {class_name}.{method_name}")
            elif calib.is_default_like():
                errors.append(f"Placeholder calibration: {class_name}.{method_name}")
        
        # Return structured result
        if errors:
            return ValidationResult(
                is_valid=False,
                severity="ERROR",
                message="; ".join(errors),
                context={"failing_checks": errors}
            )
        
        return ValidationResult(
            is_valid=True,
            severity="INFO",
            message="All contracts satisfied",
            context={"sequence_length": len(self.method_sequence)}
        )

# Usage in executor:
class AdvancedDataFlowExecutor(...):
    def __init__(self, ...):
        self.executor = method_executor
        self.config = config
        self.signal_registry = signal_registry
        
        # Create contract and validate
        contract = ExecutorContract(
            method_executor=method_executor,
            config=config,
            signal_registry=signal_registry,
            method_sequence=self._get_method_sequence()
        )
        
        result = contract.validate()
        if not result.is_valid:
            raise RuntimeError(f"Contract violation: {result.message}")
```

---

### 5.4 Structured Argument Context Management

**Current**: Mutable dict with reset pattern

**Proposal**: Immutable context snapshots
```python
@dataclass(frozen=True)
class ExecutionContext:
    """Immutable execution context snapshot."""
    doc: Any
    current_data: Any
    signals: dict[str, Any] | None
    consumption_proof: SignalConsumptionProof | None
    metadata: dict[str, Any]
    
    def for_method(self, method_key: str) -> 'ExecutionContext':
        """Create context variant for specific method."""
        return ExecutionContext(
            doc=self.doc,
            current_data=self.current_data,
            signals=self.signals,
            consumption_proof=self.consumption_proof,
            metadata={
                **self.metadata,
                'current_method': method_key,
                'timestamp_utc': time.time()
            }
        )

# Usage:
context = ExecutionContext(
    doc=doc,
    current_data=doc.raw_text,
    signals=signals,
    consumption_proof=consumption_proof,
    metadata={'policy_area': policy_area}
)

for class_name, method_name in method_sequence:
    method_context = context.for_method(f"{class_name}.{method_name}")
    kwargs = self._prepare_arguments_with_context(method_context)
    result = self.executor.execute(class_name, method_name, **kwargs)
    # Context is never modified, only derived variants created
```

---

### 5.5 Explicit Signal Contract

**Current**: Ad-hoc attribute checks with hasattr()

**Proposal**: Formal SignalPack protocol
```python
from typing import Protocol, TypedDict

class SignalPattern(TypedDict):
    pattern: str
    category: str
    confidence: float

class SignalPack(Protocol):
    """Formal contract for signal packs."""
    
    version: str
    policy_area: str
    patterns: list[SignalPattern]
    indicators: list[str]
    regex: dict[str, str]
    verbs: list[str]
    entities: list[str]
    thresholds: dict[str, float]
    
    def compute_hash(self) -> str:
        """Return SHA-256 hash of signal pack."""
        ...
    
    def get_keys_used(self) -> list[str]:
        """Return list of keys accessed during signal processing."""
        ...

# Validation in executor:
def _fetch_signals(self, policy_area: str) -> SignalPack | None:
    """Fetch signals with formal contract validation."""
    signal_pack = self.signal_registry.get(policy_area)
    
    if signal_pack is None:
        logger.warning(f"Signal pack not found for {policy_area}")
        return None
    
    # Validate protocol conformance
    required_attrs = ['version', 'policy_area', 'patterns', 'compute_hash']
    for attr in required_attrs:
        if not hasattr(signal_pack, attr):
            logger.error(f"Signal pack missing required attribute: {attr}")
            return None
    
    try:
        signal_hash = signal_pack.compute_hash()
    except Exception as e:
        logger.error(f"Signal pack compute_hash() failed: {e}")
        return None
    
    return signal_pack
```

---

### 5.6 Calibration Contract Separation

**Current**: Optional CalibrationOrchestrator, mandatory registry lookup

**Proposal**: Explicit calibration modes
```python
class CalibrationMode(Enum):
    STRICT = "strict"  # Fail if calibration missing
    LENIENT = "lenient"  # Skip uncalibrated methods
    NONE = "none"  # Ignore calibrations entirely

@dataclass(frozen=True)
class CalibrationConfig:
    mode: CalibrationMode
    skip_threshold: float = 0.3
    orchestrator: CalibrationOrchestrator | None = None

class AdvancedDataFlowExecutor(...):
    def __init__(
        self,
        ...,
        calibration_config: CalibrationConfig = CalibrationConfig(mode=CalibrationMode.STRICT)
    ):
        self.calibration_config = calibration_config
        
        # Validate based on mode
        if self.calibration_config.mode == CalibrationMode.STRICT:
            self._validate_all_calibrations()
        elif self.calibration_config.mode == CalibrationMode.LENIENT:
            self._warn_uncalibrated_methods()
    
    def execute_with_optimization(self, ...):
        # Apply calibration based on mode
        if self.calibration_config.mode == CalibrationMode.NONE:
            # Skip calibration phase entirely
            pass
        else:
            # Run calibration phase
            calibration_results = self._calibration_phase()
            
            if self.calibration_config.mode == CalibrationMode.LENIENT:
                # Skip low-scoring methods
                for method_key, result in calibration_results.items():
                    if result.final_score < self.calibration_config.skip_threshold:
                        # Skip this method
                        pass
```

---

### 5.7 Thread-Safe Deterministic Seeding

**Current**: Global random.seed() modification

**Proposal**: Thread-local RNG
```python
import threading

class DeterministicRNG:
    """Thread-safe deterministic RNG."""
    
    _local = threading.local()
    
    @classmethod
    def seed_from_context(cls, policy_unit_id: str, correlation_id: str) -> 'DeterministicRNG':
        """Create seeded RNG for execution context."""
        components = [
            str(policy_unit_id),
            str(correlation_id),
            str(threading.get_ident())  # Include thread ID
        ]
        material = "|".join(components)
        digest = hashlib.sha256(material.encode("utf-8")).digest()
        base_seed = int.from_bytes(digest[:4], byteorder="big")
        
        rng = DeterministicRNG(base_seed)
        cls._local.rng = rng  # Store thread-locally
        return rng
    
    def __init__(self, seed: int):
        self.np_rng = np.random.default_rng(seed)
        self.python_rng = random.Random(seed)
    
    @classmethod
    @contextmanager
    def for_context(cls, policy_unit_id: str, correlation_id: str):
        """Context manager for deterministic execution."""
        rng = cls.seed_from_context(policy_unit_id, correlation_id)
        try:
            yield rng
        finally:
            cls._local.rng = None

# Usage:
with DeterministicRNG.for_context(policy_unit_id, correlation_id) as rng:
    # All random operations in this scope use rng
    result = self.quantum_optimizer.select_optimal_path(
        available_methods, rng=rng
    )
```

---

### 5.8 Advanced Module Activation Guards

**Current**: Always initialized, activation conditions undocumented

**Proposal**: Conditional initialization with gating
```python
@dataclass(frozen=True)
class AdvancedModuleGates:
    """Control which frontier modules are active."""
    
    enable_quantum: bool = True
    enable_neuromorphic: bool = True
    enable_causal: bool = True
    enable_info_theory: bool = True
    enable_meta_learning: bool = True
    enable_attention: bool = True

class AdvancedDataFlowExecutor(...):
    def __init__(
        self,
        ...,
        gates: AdvancedModuleGates | None = None
    ):
        self.gates = gates or AdvancedModuleGates()
        
        # Initialize only enabled modules
        if self.gates.enable_quantum and len(method_sequence) >= 3:
            self.quantum_optimizer = QuantumExecutionOptimizer(
                num_methods=len(method_sequence)
            )
        else:
            self.quantum_optimizer = None
        
        if self.gates.enable_neuromorphic and len(method_sequence) > 1:
            self.neuromorphic_controller = NeuromorphicFlowController(
                num_stages=len(method_sequence)
            )
        else:
            self.neuromorphic_controller = None
        
        # ... etc for other modules
    
    def execute_with_optimization(self, ...):
        # Only use activated modules
        if self.quantum_optimizer:
            path = self.quantum_optimizer.select_optimal_path(...)
        
        if self.neuromorphic_controller:
            self.neuromorphic_controller.process_data_flow(...)
```

---

## 6. SUMMARY TABLE

| Aspect | Current State | Risk Level | Proposal |
|--------|--------------|-----------|----------|
| **Config Enforcement** | Enforced but dead code (line 1345) | Medium | Remove unreachable fallback |
| **Method Sequences** | Two conflicting patterns | High | Standardize CLASS attribute pattern |
| **Validation Timing** | Runtime for most executors | High | Move to construction time |
| **Factory-Executor Coupling** | Decoupled | High | Unified ExecutorBuilder factory |
| **Calibration Contracts** | Optional + mandatory = ambiguous | High | Explicit CalibrationConfig enum |
| **Signal Contracts** | Ad-hoc hasattr() checks | Medium | Formal SignalPack Protocol |
| **Argument Context** | Mutable dict with reset | Medium | Immutable context snapshots |
| **Deterministic Seeding** | Global random.seed() | Medium | Thread-local DeterministicRNG |
| **Module Activation** | Always initialized | Low | Conditional gates |
| **Error Handling** | Ad-hoc exception swallowing | Medium | Explicit error contracts |

---

## 7. CONCLUSION

The executors.py module contains sophisticated frontier paradigm implementations but suffers from **loose factory-executor coupling** and **inconsistent validation patterns**. The 30 executor subclasses diverge into two patterns, creating maintenance burden and runtime error risks.

**Priority Recommendations**:
1. **HIGH**: Implement unified ExecutorBuilder factory
2. **HIGH**: Standardize METHOD_SEQUENCE class attributes
3. **HIGH**: Separate and clarify calibration contracts
4. **MEDIUM**: Introduce immutable ExecutionContext for argument management
5. **MEDIUM**: Define formal SignalPack protocol
6. **MEDIUM**: Add thread-safe DeterministicRNG

These changes would improve **contract safety, earlier error detection, and maintainability** while preserving all existing frontier optimizations.
