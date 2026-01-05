use webnn_onnx_utils::operation_names::mapper;

#[test]
fn basic_op_mappings_work_case_insensitive() {
    let m = mapper();

    assert_eq!(m.webnn_to_onnx("matmul"), Some("MatMul"));
    assert_eq!(m.webnn_to_onnx("MaTmUl"), Some("MatMul"));

    assert_eq!(m.onnx_to_webnn("MatMul"), Some("matmul"));
    assert_eq!(m.onnx_to_webnn("matmul"), Some("matmul"));
}
