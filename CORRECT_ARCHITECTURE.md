# Correct Architecture - Questionnaire Data Flow

**Date**: 2025-11-06  
**Status**: Architecture clarified, incorrect changes reverted

---

## User's Clarification on Correct Architecture

### ❌ What Was WRONG (My Mistake)

I incorrectly implemented questionnaire data flowing to **PolicyProcessor**:

```python
# ❌ WRONG - What I did
processor = IndustrialPolicyProcessor(questionnaire_data=questionnaire)
```

**Problem**: PolicyProcessor should NOT have direct access to questionnaire data.

### ✅ What Is CORRECT (User's Direction)

**User's Key Points:**

1. **"Policy processor CAN NOT HAVE ACCESS TO THE QUESTIONARY"**
   - PolicyProcessor is a script with various classes, functions, and methods
   - We only use a portion of its methods via executors
   - PolicyProcessor itself should NOT receive questionnaire data

2. **"It is the orchestrator which manages this access"**
   - Orchestrator controls questionnaire loading and distribution
   - Single source of truth

3. **"Executors receive data from the questionnaire monolith to enrich the process"**
   - EXECUTORS (not PolicyProcessor) should receive questionnaire data
   - Executors use questionnaire patterns, verbs, entities to enrich processing
   - "Check the files !!!! and recent pull request we have created a channel"

---

## Correct Architecture Flow

```
┌─────────────────────────────────────────────────────────┐
│                   ORCHESTRATOR                          │
│  - Loads questionnaire_monolith.json ONCE              │
│  - Manages access to questionnaire data                │
└──────────────┬──────────────────────────────────────────┘
               │
               ├─→ Questionnaire Data
               │
               ↓
    ┌──────────────────────┐
    │     EXECUTORS        │  ✅ Receive questionnaire data
    │  (D1Q1, D1Q2, ...)   │  ✅ Use patterns/verbs/entities
    │                      │  ✅ Enrich processing
    └──────────┬───────────┘
               │
               ├─→ Call PolicyProcessor methods
               │
               ↓
    ┌──────────────────────┐
    │  PolicyProcessor     │  ❌ Does NOT receive questionnaire
    │  - process()         │  ✅ Only provides methods
    │  - _match_patterns() │  ✅ Called BY executors
    │  - etc.              │
    └──────────────────────┘
```

---

## Key Architectural Principles

### 1. Orchestrator Manages Access
- Orchestrator loads questionnaire ONCE via factory
- Orchestrator distributes to appropriate consumers
- Single source of truth

### 2. Executors Are Enriched
- Executors receive questionnaire data
- Use patterns, regexes, verbs, entities from questionnaire
- Enrich their processing with this rich data

### 3. PolicyProcessor Is a Service
- PolicyProcessor provides methods/functions
- Does NOT need direct questionnaire access
- Methods are called BY executors, not standalone

### 4. No Dual Factories
- Only `src/saaaaaa/core/orchestrator/factory.py` exists
- Old `orchestrator/factory.py` deleted

---

## What Needs to Be Implemented

### Option 1: Pass Questionnaire to Executor Instances

```python
# In Orchestrator initialization
class Orchestrator:
    def __init__(self, monolith=None, ...):
        # Load questionnaire
        questionnaire_data = monolith  # Pre-loaded
        
        # Create executors WITH questionnaire
        self.executors = {
            "D1-Q1": D1Q1_Executor(questionnaire_data),
            "D1-Q2": D1Q2_Executor(questionnaire_data),
            # ... all 30 executors
        }
```

### Option 2: Pass Questionnaire at Execution Time

```python
# In executor execute method
class D1Q1_Executor(AdvancedDataFlowExecutor):
    def execute(self, doc, method_executor, questionnaire_data=None):
        # Use questionnaire_data to enrich processing
        patterns = questionnaire_data.get('patterns', [])
        verbs = questionnaire_data.get('verbs', [])
        # ...
```

### Option 3: Store Questionnaire in MethodExecutor Context

```python
# In MethodExecutor
class MethodExecutor:
    def __init__(self, questionnaire_data=None):
        self.questionnaire_data = questionnaire_data
        # Available to executors via method_executor.questionnaire_data
```

---

## User Mentioned Recent PR

User said: "check the files !!!! and recent pull request we have created a channel"

This suggests there may already be a channel implementation I should review. Need to:
1. Check recent PRs for questionnaire channel implementation
2. Understand existing pattern
3. Follow that pattern

---

## Hard-Coded Method Sequences Issue

User also mentioned:

> "30 executor classes each with 15-35 hard-coded method sequences. Any change to method signatures breaks everything. Zero flexibility."

**Current State:**
```python
method_sequence = [
    ('IndustrialPolicyProcessor', 'process'),
    ('IndustrialPolicyProcessor', '_match_patterns_in_sentences'),
    # ... 18 more hard-coded calls
]
```

**Problem**: Brittle, inflexible, hard to maintain

**Potential Solutions:**
1. Dynamic method discovery
2. Configuration-driven sequences
3. Registry-based method lookup
4. Meta-programming for signature adaptation

This is separate from questionnaire flow but important for overall architecture.

---

## Summary

### ✅ What Was Fixed
- Reverted all incorrect PolicyProcessor questionnaire injection
- Deleted old factory (`orchestrator/factory.py`)
- Removed incorrect tests and documentation

### ⚠️ What Needs Implementation
- Questionnaire data flow to EXECUTORS
- Check recent PR for existing channel pattern
- Follow user's existing implementation pattern

### 🔍 What Needs Investigation
- Review recent PRs for questionnaire channel
- Understand existing pattern
- Implement consistently with user's approach

---

**Status**: Incorrect changes reverted, awaiting direction on correct implementation pattern  
**Commit**: 419c454
