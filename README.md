# webnn-onnx-utils

Shared ONNX/WebNN conversion utilities used by:

- `rustnn` (WebNN → ONNX)
- `webnn-graph` (ONNX → WebNN)

This crate is intentionally small and modular. It provides:

- data type mapping
- operation name mapping
- attribute parsing/building helpers
- tensor initializer data helpers
- identifier sanitization
- (later) shape inference utilities

## Protobufs

ONNX protobufs are compiled with `prost-build` from `protos/onnx/onnx.proto3`.
