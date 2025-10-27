---
type: "manual"
---

Rule 3: The Imperative of Deterministic Execution & Explicit Failure

Philosophical Grounding

This rule is the direct implementation of the Prime Directives: No Graceful Degradation and Deterministic Reproducibility over Throughput Opportunism. The system is an instrument of precision. Its outputs must be trustworthy, repeatable, and auditable. Therefore, all forms of ambiguity, silent failure, and non-determinism are treated as hostile elements to be controlled or eliminated. Every process must either fulfill its contract perfectly or fail loudly and explicitly.

Mandates of Rule 3

3.1: The Mandate of Explicit Contracts: Every function, method, or transformation must operate under an explicit contract. Its docstring or accompanying documentation must clearly state its preconditions (what must be true for it to run), its postconditions (what will be true after it successfully completes), and its invariants (what remains unchanged during its execution). Code should use assertions to enforce these contracts at runtime.

3.2: Absolute Prohibition of Silent Failure and Graceful Degradation: A function or process is forbidden from "partially" succeeding or silently returning a fallback value. If a process cannot fulfill its entire contract, it must raise an explicit, informative, and diagnosable exception. The system must fail in a way that makes the source and nature of the error immediately obvious.

3.3: Strict Control of All Non-Determinism: Any element that could introduce variability—including random number generation, floating-point arithmetic, or multi-threading—must be strictly controlled. Random processes must be seeded from a deterministic source (e.g., seed_factory.py), and the seed must be logged. All operations must strive for bit-for-bit reproducibility to ensure that a given input will always produce the exact same output.

3.4: The Mandate of Structural Observability: Traceability and logging are not afterthoughts; they are structural requirements. All new features must be designed from the ground up to be fully observable. The execution path, critical intermediate values, and all decisions made by the logic must be logged in a structured format that allows for the complete and unambiguous reconstruction of any analytical process.