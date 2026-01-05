use webnn_onnx_utils::data_types::{onnx_to_webnn, webnn_to_onnx, DataType};

#[test]
fn round_trip_dtype_codes() {
    let all = [
        DataType::Float32,
        DataType::Float16,
        DataType::Int32,
        DataType::Uint32,
        DataType::Int64,
        DataType::Uint64,
        DataType::Int8,
        DataType::Uint8,
    ];

    for dt in all {
        let proto = webnn_to_onnx(dt.clone());
        let back = onnx_to_webnn(proto as i32).unwrap();
        assert_eq!(dt, back);
    }
}
