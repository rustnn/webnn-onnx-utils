use webnn_onnx_utils::operation_names::mapper;

#[test]
fn basic_op_mappings_work_case_insensitive() {
    let m = mapper();

    assert_eq!(m.webnn_to_onnx("matmul"), Some("MatMul"));
    assert_eq!(m.webnn_to_onnx("MaTmUl"), Some("MatMul"));

    assert_eq!(m.onnx_to_webnn("MatMul"), Some("matmul"));
    assert_eq!(m.onnx_to_webnn("matmul"), Some("matmul"));
}

#[test]
fn test_matrix_operations() {
    let m = mapper();

    assert_eq!(m.webnn_to_onnx("matmul"), Some("MatMul"));
    assert_eq!(m.webnn_to_onnx("gemm"), Some("Gemm"));

    assert_eq!(m.onnx_to_webnn("MatMul"), Some("matmul"));
    assert_eq!(m.onnx_to_webnn("Gemm"), Some("gemm"));
}

#[test]
fn test_convolution_operations() {
    let m = mapper();

    assert_eq!(m.webnn_to_onnx("conv2d"), Some("Conv"));
    assert_eq!(m.webnn_to_onnx("convTranspose2d"), Some("ConvTranspose"));

    assert_eq!(m.onnx_to_webnn("Conv"), Some("conv2d"));
    assert_eq!(m.onnx_to_webnn("ConvTranspose"), Some("convTranspose2d"));
}

#[test]
fn test_pooling_operations() {
    let m = mapper();

    assert_eq!(m.webnn_to_onnx("averagePool2d"), Some("AveragePool"));
    assert_eq!(m.webnn_to_onnx("maxPool2d"), Some("MaxPool"));
    assert_eq!(
        m.webnn_to_onnx("globalAveragePool"),
        Some("GlobalAveragePool")
    );

    assert_eq!(m.onnx_to_webnn("AveragePool"), Some("averagePool2d"));
    assert_eq!(m.onnx_to_webnn("MaxPool"), Some("maxPool2d"));
}

#[test]
fn test_activation_operations() {
    let m = mapper();

    assert_eq!(m.webnn_to_onnx("relu"), Some("Relu"));
    assert_eq!(m.webnn_to_onnx("sigmoid"), Some("Sigmoid"));
    assert_eq!(m.webnn_to_onnx("tanh"), Some("Tanh"));
    assert_eq!(m.webnn_to_onnx("softmax"), Some("Softmax"));
    assert_eq!(m.webnn_to_onnx("gelu"), Some("Gelu"));

    assert_eq!(m.onnx_to_webnn("Relu"), Some("relu"));
    assert_eq!(m.onnx_to_webnn("Sigmoid"), Some("sigmoid"));
}

#[test]
fn test_elementwise_operations() {
    let m = mapper();

    assert_eq!(m.webnn_to_onnx("add"), Some("Add"));
    assert_eq!(m.webnn_to_onnx("sub"), Some("Sub"));
    assert_eq!(m.webnn_to_onnx("mul"), Some("Mul"));
    assert_eq!(m.webnn_to_onnx("div"), Some("Div"));

    assert_eq!(m.onnx_to_webnn("Add"), Some("add"));
    assert_eq!(m.onnx_to_webnn("Mul"), Some("mul"));
}

#[test]
fn test_unary_operations() {
    let m = mapper();

    assert_eq!(m.webnn_to_onnx("abs"), Some("Abs"));
    assert_eq!(m.webnn_to_onnx("exp"), Some("Exp"));
    assert_eq!(m.webnn_to_onnx("log"), Some("Log"));
    assert_eq!(m.webnn_to_onnx("sqrt"), Some("Sqrt"));

    assert_eq!(m.onnx_to_webnn("Abs"), Some("abs"));
    assert_eq!(m.onnx_to_webnn("Exp"), Some("exp"));
}

#[test]
fn test_reduction_operations() {
    let m = mapper();

    assert_eq!(m.webnn_to_onnx("reduceSum"), Some("ReduceSum"));
    assert_eq!(m.webnn_to_onnx("reduceMean"), Some("ReduceMean"));
    assert_eq!(m.webnn_to_onnx("reduceMax"), Some("ReduceMax"));
    assert_eq!(m.webnn_to_onnx("reduceMin"), Some("ReduceMin"));

    assert_eq!(m.onnx_to_webnn("ReduceSum"), Some("reduceSum"));
    assert_eq!(m.onnx_to_webnn("ReduceMean"), Some("reduceMean"));
}

#[test]
fn test_comparison_operations() {
    let m = mapper();

    assert_eq!(m.webnn_to_onnx("equal"), Some("Equal"));
    assert_eq!(m.webnn_to_onnx("greater"), Some("Greater"));
    assert_eq!(m.webnn_to_onnx("lesser"), Some("Less"));

    assert_eq!(m.onnx_to_webnn("Equal"), Some("equal"));
    assert_eq!(m.onnx_to_webnn("Greater"), Some("greater"));
}

#[test]
fn test_logical_operations() {
    let m = mapper();

    assert_eq!(m.webnn_to_onnx("logicalAnd"), Some("And"));
    assert_eq!(m.webnn_to_onnx("logicalOr"), Some("Or"));
    assert_eq!(m.webnn_to_onnx("logicalNot"), Some("Not"));

    assert_eq!(m.onnx_to_webnn("And"), Some("logicalAnd"));
    assert_eq!(m.onnx_to_webnn("Or"), Some("logicalOr"));
}

#[test]
fn test_tensor_manipulation_operations() {
    let m = mapper();

    assert_eq!(m.webnn_to_onnx("concat"), Some("Concat"));
    assert_eq!(m.webnn_to_onnx("reshape"), Some("Reshape"));
    assert_eq!(m.webnn_to_onnx("transpose"), Some("Transpose"));
    assert_eq!(m.webnn_to_onnx("squeeze"), Some("Squeeze"));
    assert_eq!(m.webnn_to_onnx("unsqueeze"), Some("Unsqueeze"));

    assert_eq!(m.onnx_to_webnn("Concat"), Some("concat"));
    assert_eq!(m.onnx_to_webnn("Reshape"), Some("reshape"));
}

#[test]
fn test_normalization_operations() {
    let m = mapper();

    assert_eq!(
        m.webnn_to_onnx("batchNormalization"),
        Some("BatchNormalization")
    );
    assert_eq!(
        m.webnn_to_onnx("layerNormalization"),
        Some("LayerNormalization")
    );

    assert_eq!(
        m.onnx_to_webnn("BatchNormalization"),
        Some("batchNormalization")
    );
    assert_eq!(
        m.onnx_to_webnn("LayerNormalization"),
        Some("layerNormalization")
    );
}

#[test]
fn test_missing_operations() {
    let m = mapper();

    assert_eq!(m.webnn_to_onnx("nonexistent"), None);
    assert_eq!(m.onnx_to_webnn("Nonexistent"), None);
}
