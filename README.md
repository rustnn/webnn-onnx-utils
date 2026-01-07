# webnn-onnx-utils

Shared ONNX/WebNN conversion utilities used by:

- `rust-webnn-graph` (WebNN → ONNX)
- `webnn-wg` (ONNX → WebNN)

This crate is intentionally small and modular. It provides:

- **Data type mapping** - Bidirectional conversion between WebNN and ONNX data types (Float32, Float16, Int32, Int64, etc.)
- **Operation name mapping** - 90+ operation mappings between WebNN and ONNX (matmul↔MatMul, conv2d↔Conv, etc.)
- **Attribute parsing/building** - Type-safe attribute handling for ONNX NodeProto (int, float, string, arrays)
- **Tensor data handling** - Conversion between ONNX TensorProto and typed data (with all data type support)
- **Identifier sanitization** - WebNN DSL-compatible identifier generation
- **Shape inference** - Comprehensive shape inference for common operations (matmul, transpose, reduce, concat, etc.)

## Features

### Data Types

All common ONNX and WebNN data types are supported:
- Float32, Float16
- Int8, Int32, Int64
- Uint8, Uint32, Uint64

### Operations

90+ operations mapped including:
- Matrix operations (MatMul, Gemm)
- Convolutions (Conv, ConvTranspose)
- Pooling (AveragePool, MaxPool, GlobalAveragePool)
- Activations (Relu, Sigmoid, Tanh, Softmax, Gelu)
- Elementwise (Add, Sub, Mul, Div)
- Reductions (ReduceSum, ReduceMean, ReduceMax, etc.)
- Tensor manipulation (Concat, Reshape, Transpose, Squeeze, Unsqueeze)
- And many more...

### Shape Inference

Shape inference context with support for:
- Unary and binary operations with broadcasting
- Matrix multiplication (2D and batched)
- Transpose with permutation
- Reduction operations (with keepdims support)
- Concat along any axis
- Reshape (with -1 inference)
- Squeeze and Unsqueeze
- Dynamic dimensions with overrides

## Usage

```rust
use webnn_onnx_utils::{
    data_types::{DataType, webnn_to_onnx, onnx_to_webnn},
    operation_names::mapper,
    attributes::{AttrBuilder, AttrParser},
    tensor_data::TensorData,
    shape_inference::{ShapeInferenceContext, TensorShape},
};

// Data type conversion
let onnx_type = webnn_to_onnx(DataType::Float32);
let webnn_type = onnx_to_webnn(1).unwrap(); // 1 = FLOAT

// Operation mapping
let m = mapper();
assert_eq!(m.webnn_to_onnx("matmul"), Some("MatMul"));
assert_eq!(m.onnx_to_webnn("MatMul"), Some("matmul"));

// Attribute building
let attrs = AttrBuilder::new()
    .add_int("axis", 1)
    .add_float("epsilon", 1e-5)
    .add_ints("perm", vec![0, 2, 1, 3])
    .build();

// Tensor data
let data = TensorData::scalar(DataType::Float32, 3.14);
let proto = data.to_tensor_proto("weight".to_string(), DataType::Float32, vec![1]);

// Shape inference
let mut ctx = ShapeInferenceContext::new();
ctx.set_shape("a".to_string(), TensorShape::from_static(vec![2, 3, 4]));
ctx.set_shape("b".to_string(), TensorShape::from_static(vec![2, 4, 5]));
let result = ctx.infer_matmul("a", "b").unwrap();
// result shape: [2, 3, 5]
```

## Testing

Comprehensive test coverage with 57+ tests:

```bash
cargo test
```

All tests pass covering:
- Data type round-trip conversions
- Operation name mappings (case-insensitive)
- Attribute parsing and building (all types)
- Tensor data operations
- Shape inference for all supported operations
- Broadcasting rules
- Identifier sanitization

## Protobufs

ONNX protobufs are compiled with `prost-build` from `protos/onnx/onnx.proto3`.

## License

Apache-2.0
