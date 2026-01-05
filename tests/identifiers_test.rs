use webnn_onnx_utils::identifiers::{is_valid_webnn_identifier, sanitize_for_webnn};

#[test]
fn sanitize_basic() {
    assert_eq!(sanitize_for_webnn("a::b:c.d/e"), "a__b_c_d_e");
}

#[test]
fn validate_basic() {
    assert!(is_valid_webnn_identifier("_ok1"));
    assert!(!is_valid_webnn_identifier("1nope"));
    assert!(!is_valid_webnn_identifier("has-dash"));
}
