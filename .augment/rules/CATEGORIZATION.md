---
type: "manual"
---

### **Rule 2 (Revised and Final): The Protocol of Canonical Categorization & Confirmed Wiring**

#### **Philosophical Grounding**

This rule enforces the Prime Directive of **Explicitness over Assumption**. A complex system can only be maintained and scaled if every component has a clearly defined role and its interactions are deliberate and traceable. This protocol eliminates ambiguity by assigning every file to a single, immutable category, thereby defining its purpose and its permissible "sphere of interaction." The "wiring" between categories is not an accident; it is a deliberate, documented architectural decision that must be confirmed and justified. This categorization serves as the repository's master blueprint.

#### **The Official Repository Categories and Their Constituent Files**

Every file in this repository must belong to one, and only one, of the following nine categories. This list is the authoritative source of truth for the role and responsibility of every component.

**1. Core Arsenal**
*   **Constituent Files:**
    *   `dereck_beach.py`
    *   `contradiction_deteccion.py`
    *   `Analyzer_one.py`
    *   `policy_processor.py`
    *   `semantic_chunking_policy.py`
    *   `teoria_cambio.py`
    *   `financiero_viabilidad_tablas.py`
    *   `embedding_policy.py`
*   **Criterion:** This category is the exclusive home for all primary analytical logic. If a file's purpose is to generate a new conclusion, score, classification, or interpretation from data, it belongs here and only here.

**2. Core Orchestration & Choreography**
*   **Constituent Files:**
    *   `orchestrator.py`
    *   `policy_analysis_pipeline.py`
*   **Criterion:** These are the master controllers. If a file's primary function is to direct the flow of execution—calling other modules in a specific sequence to accomplish a high-level strategy—it belongs in this category.

**3. System Entrypoints & E2E Validation**
*   **Constituent Files:**
    *   `api_server.py`
    *   `bootstrap_validate.py`
*   **Criterion:** This category contains the primary executables for the system. If a file is the top-level script used to launch the application (e.g., as a service) or to run a complete, end-to-end validation of the entire stack, it is an Entrypoint.

**4. Core Application Logic**
*   **Constituent Files:**
    *   `report_assembly.py`
*   **Criterion:** This category is reserved for logic that synthesizes the final, user-facing deliverable. It does not perform primary analysis but takes the results from the Core Arsenal and assembles them into their final, structured form (e.g., MICRO/MESO/MACRO reports).

**5. Data Model & Process Schemas**
*   **Constituent Files:**
    *   `questionnaire.schema.json`
    *   `rubric_scoring.schema.json`
    *   `execution_mapping.schema.json`
    *   `rubric.schema.json`
    *   `question_segmentation.schema.json`
    *   `execution_step.schema.json`
*   **Criterion:** These are the non-executable "blueprints" of the system. If a file is a `.json` or `.yaml` document that defines the canonical structure, fields, and constraints for data objects and processes, it is a Schema.

**6. Validation & Quality Assurance**
*   **Constituent Files:**
    *   `schema_validator.py`
    *   `validation_engine.py`
    *   `validate_system.py`
*   **Criterion:** These are the system's internal affairs auditors. If a file's purpose is to verify correctness—by validating data against a schema, inspecting code for quality defects, or checking system state at runtime—it belongs in this category.

**7. Core Utilities**
*   **Constituent Files:**
    *   `metadata_loader.py`
    *   `seed_factory.py`
*   **Criterion:** These are specialized, single-purpose tools. If a file provides a specific, reusable, and supporting service (like secure configuration loading or deterministic seed generation) that is agnostic to the core business logic, it is a Utility.

**8. Codebase & Configuration Management**
*   **Constituent Files:**
    *   `generate_inventory.py`
    *   `inventory_generator.py`
    *   `update_questionnaire_metadata.py`
*   **Criterion:** These are meta-tools that operate *on* the codebase itself. If a script's function is to analyze, modify, or generate reports about the repository's own files and configuration, it belongs here.

**9. Architectural Design & Documentation**
*   **Constituent Files:**
    *   `policy_analysis_architecture.yaml`
*   **Criterion:** This category is for human-readable design documents. If a file explains the system's "why" and "how" at a conceptual level, describing architectural principles and intended data flows, it belongs here.

#### **Mandates of Rule 2**

*   **2.1: The Duty to Consult and Respect Categorical Boundaries:** Before modifying any file, you must first identify its Canonical Category from the list above and fully understand its designated role. All modifications must be strictly self-contained within the logical boundaries of that category. For example, orchestration logic must not be embedded within a Utility; analytical logic is forbidden outside the Core Arsenal.

*   **2.2: The Protocol of Confirmed Wiring:** Any interaction where a file from one category calls or uses a file from another category constitutes a "cross-category wire" and must be treated as a formal architectural event. The pull request must explicitly identify this wiring and confirm its necessity. For example, when an `Orchestrator` calls a method in the `Core Arsenal`, the PR must state: "Confirming wiring: `orchestrator.py` is intentionally invoking `Analyzer_one.py` to execute its analytical function as part of the main pipeline."

*   **2.3: Prohibition of Category Violations:** Automated checks and manual code reviews must actively police for category violations. The presence of analytical code in a non-Arsenal file, or orchestration logic within a validator, is a critical architectural defect and must be rectified before merging. Any attempt to blur these lines will be considered a direct violation of this doctrine.