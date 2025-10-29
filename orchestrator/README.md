# Orchestrator Module

## Overview

The Orchestrator is the central coordinator for the policy evaluation pipeline. It manages the complete end-to-end execution across 11 distinct phases (0-10), ensuring proper sequencing, resource management, and comprehensive reporting.

## Architecture

### Separation of Concerns

As detailed in `ARQUITECTURA_ORQUESTADOR_COREOGRAFO.md`, the system follows a clear separation:

- **Orchestrator**: High-level pipeline coordination
  - Knows all 305 questions
  - Manages global state
  - Coordinates phases sequentially
  - Distributes work to workers
  - Aggregates results at multiple levels

- **Choreographer** (separate module): Individual question execution
  - Executes ONE question at a time
  - Interprets method DAG
  - Coordinates sync/async methods
  - Returns results to Orchestrator

## Pipeline Phases

The Orchestrator coordinates 11 distinct phases:

### Phase 0: Validation
- Validates configuration integrity
- Loads questionnaire monolith (300 questions)
- Loads method catalog (593 method packages)
- Verifies data contracts

### Phase 1: Ingestion
- Preprocesses input document
- Extracts text and tables
- Prepares document for analysis

### Phase 2: Micro Question Execution
- Executes 300 micro questions in parallel
- Configurable parallelism (default: 50 workers)
- Individual question timeouts
- Retry logic for failed questions
- Progress tracking

### Phase 3: Scoring
- Applies scoring algorithms to question results
- Calculates confidence levels
- Generates scored results

### Phase 4: Dimension Aggregation
- Aggregates into 60 dimensions (6 dimensions × 10 areas)

### Phase 5: Area Aggregation
- Aggregates into 10 policy areas

### Phase 6: Cluster Aggregation
- Aggregates into 4 clusters (MESO questions)

### Phase 7: Macro Evaluation
- Performs holistic evaluation
- Generates overall score and confidence

### Phase 8: Recommendations
- Generates recommendations at all levels

### Phase 9: Report Assembly
- Assembles complete report with all levels

### Phase 10: Output Formatting
- Generates multiple output formats (JSON, HTML, PDF, Excel)

## Usage

### Basic Usage

```python
from orchestrator import Orchestrator

# Create orchestrator with default configuration
orchestrator = Orchestrator()

# Process a document
report = orchestrator.process_document("path/to/document.pdf")

# Access results at different levels
print(f"Overall score: {report.macro_score.overall_score}")
print(f"Clusters: {len(report.cluster_scores)}")
print(f"Areas: {len(report.area_scores)}")
print(f"Questions: {len(report.question_results)}")
```

### Custom Configuration

```python
from orchestrator import Orchestrator, OrchestratorConfig

# Create custom configuration
config = OrchestratorConfig(
    max_workers=25,                    # Limit parallelism
    default_question_timeout=120.0,    # 2 minutes per question
    min_completion_rate=0.85,          # Allow 15% failure rate
    log_level="DEBUG",                 # Detailed logging
)

orchestrator = Orchestrator(config=config)
report = orchestrator.process_document("document.pdf")
```

### Monitoring Progress

```python
orchestrator = Orchestrator()

# Start processing in separate thread
import threading
thread = threading.Thread(
    target=orchestrator.process_document,
    args=("document.pdf",)
)
thread.start()

# Monitor progress
while thread.is_alive():
    status = orchestrator.get_processing_status()
    print(f"Phase: {status.current_phase.name}")
    print(f"Progress: {status.completion_percentage:.1f}%")
    print(f"Questions: {status.questions_completed}/{status.questions_total}")
    time.sleep(5)
```

### Abort Control

```python
orchestrator = Orchestrator()

# Start processing
thread = threading.Thread(
    target=orchestrator.process_document,
    args=("document.pdf",)
)
thread.start()

# Request abort if needed
if some_condition:
    orchestrator.request_abort()
    # Processing will stop gracefully at next phase boundary
```

### Validation Only

```python
orchestrator = Orchestrator()

# Validate configuration without processing
try:
    orchestrator.validate_configuration()
    print("Configuration is valid")
except ValidationError as e:
    print(f"Configuration error: {e}")
```

## Configuration Options

### OrchestratorConfig

```python
@dataclass
class OrchestratorConfig:
    # Pool configuration
    max_workers: int = 50              # Max parallel questions
    min_workers: int = 10              # Min workers to keep warm
    
    # Timeouts (seconds)
    default_question_timeout: float = 180.0   # 3 minutes
    complex_question_timeout: float = 300.0   # 5 minutes
    global_timeout: float = 3600.0            # 1 hour total
    
    # Retry configuration
    max_question_retries: int = 3
    retry_backoff_factor: float = 2.0  # Exponential backoff
    
    # Tolerance settings
    min_completion_rate: float = 0.9   # Require 90% success
    allow_partial_report: bool = True  # Generate partial reports
    
    # Resource limits
    memory_limit_per_worker: str = "2GB"
    cpu_cores_per_worker: int = 1
    
    # Monitoring
    progress_report_interval: int = 30  # seconds
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    # Phase control
    enable_phase_validation: bool = True
    abort_on_critical_failure: bool = True
```

## Metrics and Status

### Processing Status

```python
status = orchestrator.get_processing_status()

print(f"Current phase: {status.current_phase.name}")
print(f"Progress: {status.completion_percentage:.1f}%")
print(f"Questions completed: {status.questions_completed}")
print(f"Elapsed time: {status.elapsed_time_seconds:.1f}s")
print(f"Estimated remaining: {status.estimated_time_remaining_seconds:.1f}s")
```

### Detailed Metrics

```python
metrics = orchestrator.get_metrics()

print(f"Total questions: {metrics['total_questions']}")
print(f"Completed: {metrics['completed_questions']}")
print(f"Progress: {metrics['progress']:.1%}")

# Phase-level metrics
for phase_name, phase_data in metrics['phase_metrics'].items():
    print(f"\n{phase_name}:")
    print(f"  Status: {phase_data['status']}")
    print(f"  Duration: {phase_data['duration']:.2f}s")
    print(f"  Success rate: {phase_data['success_rate']:.1%}")
```

## Error Handling

### Phase-Level Errors

Each phase execution is wrapped with error handling:

- Errors are logged with full context
- Phase metrics track failures
- Processing can continue if error is recoverable
- Critical errors abort the pipeline

### Question-Level Errors

Individual question failures:

- Automatic retry with exponential backoff
- Failed questions marked with error status
- Processing continues with other questions
- Minimum completion rate enforced

### Example Error Handling

```python
from orchestrator import Orchestrator, OrchestratorError, ValidationError

orchestrator = Orchestrator()

try:
    report = orchestrator.process_document("document.pdf")
except ValidationError as e:
    print(f"Configuration error: {e}")
except OrchestratorError as e:
    print(f"Processing error: {e}")
    # Check partial results
    status = orchestrator.get_processing_status()
    print(f"Completed {status.questions_completed} questions")
```

## Integration with Choreographer

The Orchestrator delegates individual question execution to Choreographer instances:

```python
# In orchestrator_core.py
def _execute_single_question(
    self,
    question_global: int,
    preprocessed_doc: PreprocessedDocument
) -> QuestionResult:
    """Execute a single question via Choreographer."""
    # Create choreographer instance
    choreographer = Choreographer(
        method_catalog=self.method_catalog,
        monolith=self.monolith,
        config=self.choreographer_config
    )
    
    # Execute the question
    result = choreographer.execute_question(
        question_global=question_global,
        preprocessed_doc=preprocessed_doc
    )
    
    return result
```

## Testing

The module includes comprehensive tests:

```bash
# Run all orchestrator tests
python -m unittest tests.test_orchestrator -v

# Run specific test class
python -m unittest tests.test_orchestrator.TestOrchestratorValidation -v

# Run demo
python examples/orchestrator_demo.py
```

### Test Coverage

- **Initialization**: Configuration, state setup
- **Validation**: Monolith loading, catalog validation
- **Phase execution**: Metrics tracking, error handling
- **Question execution**: Result generation, base slot calculation
- **State management**: Progress tracking, metrics collection
- **Abort control**: Graceful shutdown
- **Configuration**: Default values, customization

All 20 tests passing with 100% success rate.

## Best Practices

### 1. Configuration Management

```python
# Use configuration for different environments
dev_config = OrchestratorConfig(
    max_workers=5,
    log_level="DEBUG",
)

prod_config = OrchestratorConfig(
    max_workers=100,
    log_level="INFO",
    abort_on_critical_failure=True,
)
```

### 2. Resource Management

```python
# Adjust workers based on available resources
import psutil

available_cores = psutil.cpu_count()
config = OrchestratorConfig(
    max_workers=min(50, available_cores * 2)
)
```

### 3. Monitoring Integration

```python
# Integrate with monitoring systems
import time

orchestrator = Orchestrator()
while True:
    status = orchestrator.get_processing_status()
    
    # Send to monitoring system
    send_metric("orchestrator.progress", status.progress)
    send_metric("orchestrator.questions_completed", status.questions_completed)
    
    if status.is_complete:
        break
    
    time.sleep(30)
```

### 4. Error Recovery

```python
# Implement retry logic for transient failures
max_retries = 3
for attempt in range(max_retries):
    try:
        report = orchestrator.process_document("document.pdf")
        break
    except OrchestratorError as e:
        if attempt == max_retries - 1:
            raise
        print(f"Attempt {attempt + 1} failed, retrying...")
        time.sleep(5 * (2 ** attempt))  # Exponential backoff
```

## Performance Considerations

### Parallelism

- Default 50 workers provides good balance
- Adjust based on CPU cores and memory
- Monitor system resources during execution

### Timeouts

- Set appropriate timeouts for your environment
- Complex questions may need longer timeouts
- Balance between thoroughness and responsiveness

### Memory Management

- Each worker maintains state for one question
- Preprocessed document shared across workers
- Results accumulated in memory during processing

## Future Enhancements

### Planned Features

1. **Distributed Execution**: Run workers on multiple machines
2. **Checkpoint/Resume**: Save state and resume from checkpoints
3. **Dynamic Scaling**: Adjust workers based on load
4. **Streaming Results**: Stream results as they complete
5. **Priority Queuing**: Prioritize certain questions
6. **Resource Limits**: Hard memory and CPU limits per worker

## See Also

- `ARQUITECTURA_ORQUESTADOR_COREOGRAFO.md`: Architecture documentation
- `orchestrator_types.py`: Data type definitions
- `orchestrator_core.py`: Core implementation
- `examples/orchestrator_demo.py`: Usage examples
- `tests/test_orchestrator.py`: Test suite

## Support

For issues or questions:
1. Check the test suite for usage examples
2. Review the architecture documentation
3. Run the demo script for interactive exploration
4. Consult logs for detailed execution information
