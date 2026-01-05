# AGENTS.md

## Project: webnn-onnx-utils

### Purpose

`webnn-onnx-utils` is a shared Rust library that centralizes common logic for
bidirectional conversion between **WebNN** and **ONNX**.

It exists to eliminate duplicated, diverging implementations currently found in:

- `rustnn` (WebNN → ONNX) located at https://github.com/tarekziade/rustnn
- `webnn-graph` (ONNX → WebNN) located at https://github.com/tarekziade/webnn-graph

The crate intentionally focuses on **low-level, format-agnostic utilities** and
does **not** own graph-level representations, conversion pipelines, or policy.

---

## Design Goals

1. **Single source of truth**
   - Data type mappings
   - Operation name mappings
   - Attribute parsing and construction
   - Tensor / initializer data handling
   - Identifier sanitization
   - (Eventually) shape inference

2. **Bidirectional symmetry**
   - Every shared concept should support both:
     - WebNN → ONNX
     - ONNX → WebNN
   - APIs should avoid encoding direction-specific assumptions.

3. **No graph ownership**
   - This crate does not define:
     - WebNN graph IR
     - ONNX graph IR
     - Conversion orchestration
   - It only manipulates **values**, **metadata**, and **protos**.

4. **Predictable, conservative APIs**
   - Prefer explicit enums and structs over loosely typed helpers
   - Fail early on unsupported or ambiguous cases
   - Avoid “magic” inference outside clearly scoped modules

5. **Safe to share**
   - No dependency on either consumer project
   - Self-contained ONNX protobuf bindings
   - Minimal public surface area

---

## Repository Structure

```
webnn-onnx-utils/
├── AGENTS.md              # This file
├── Cargo.toml
├── build.rs               # prost-build for ONNX protobufs
├── protos/
│   └── onnx/              # Vendored ONNX .proto3 files
│       └── onnx.proto3
├── src/
│   ├── lib.rs             # Crate entry point
│   ├── error.rs           # Shared error types
│   ├── protos.rs          # Generated ONNX protobuf bindings
│   ├── data_types.rs      # WebNN ↔ ONNX data type mapping
│   ├── operation_names.rs # WebNN ↔ ONNX op name mapping
│   ├── attributes.rs      # Attribute parsing & building helpers
│   ├── tensor_data.rs     # Tensor / initializer data helpers
│   ├── identifiers.rs     # Identifier sanitization & validation
│   └── shape_inference.rs # Shape inference scaffolding (incremental)
└── tests/
    ├── data_types_test.rs
    ├── operation_names_test.rs
    └── identifiers_test.rs
```

---

## Module Responsibilities

### `data_types.rs`
- Defines the shared `DataType` enum
- Converts between WebNN data types and ONNX `TensorProto::DataType`
- Must remain exhaustive and consistent
- **High priority, correctness-critical**

### `operation_names.rs`
- Centralized bidirectional mapping between WebNN op names and ONNX op types
- Case-insensitive lookup
- Acts as the canonical registry of supported operations
- **High priority, extensibility-critical**

### `attributes.rs`
- Typed helpers for:
  - Parsing ONNX `AttributeProto` → WebNN-friendly values
  - Building ONNX `AttributeProto` from WebNN inputs
- Encodes ONNX attribute typing rules explicitly
- Avoids ad-hoc parsing in downstream crates

### `tensor_data.rs`
- Unified representation of tensor / initializer payloads
- Handles ONNX `TensorProto` fields (`raw_data`, typed fields)
- Designed to support both inline and external weights
- Incrementally expanded as more types are needed

### `identifiers.rs`
- Sanitizes ONNX identifiers for WebNN DSL compatibility
- Provides validation and unique-name helpers
- Must remain deterministic and reversible where possible

### `shape_inference.rs`
- Shared shape inference primitives
- Supports:
  - Static dimensions
  - Symbolic / dynamic dimensions with overrides
- Initially minimal; expanded over time
- Intended to replace duplicated logic in consumers

### `protos.rs` + `build.rs`
- Owns ONNX protobuf compilation via `prost`
- Consumers must not generate their own ONNX bindings
- Ensures ABI and semantic consistency across projects

---

## What Does NOT Belong Here

❌ Graph traversal logic  
❌ Model-level validation  
❌ Constant folding execution engines  
❌ WebNN JSON serialization  
❌ ONNX file I/O or packaging  
❌ Runtime execution or backends  

Those concerns live in the consuming projects.

---

## How to Extend This Crate

When adding new functionality:

1. **Confirm it is shared**
   - If only one consumer needs it, keep it local.

2. **Add bidirectional support**
   - If adding a new op, dtype, or attribute:
     - Ensure both directions are represented.

3. **Add tests**
   - Prefer table-driven tests.
   - Tests should not depend on consumer crates.

4. **Avoid breaking changes**
   - Additive changes are preferred.
   - Breaking changes require coordination with both consumers.

---

## Expected Consumers

- `rustnn`
- `webnn-wg`
- Potential future converters (e.g. TFLite, CoreML, custom DSLs)

This crate is intended to be **foundational infrastructure**.

---

## Guiding Principle

> “Fix it once, share it forever.”

If logic is duplicated in two converters, it probably belongs here.
