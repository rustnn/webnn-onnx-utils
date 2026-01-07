use webnn_onnx_utils::data_types::DataType;
use webnn_onnx_utils::tensor_data::TensorData;

#[test]
fn test_tensor_data_scalar_float32() {
    let data = TensorData::scalar(DataType::Float32, 3.14);
    assert_eq!(data.len(), 1);

    match data {
        TensorData::Float32(ref v) => assert_eq!(v[0], 3.14),
        _ => panic!("Expected Float32 variant"),
    }
}

#[test]
fn test_tensor_data_scalar_int32() {
    let data = TensorData::scalar(DataType::Int32, 42.0);
    assert_eq!(data.len(), 1);

    match data {
        TensorData::Int32(ref v) => assert_eq!(v[0], 42),
        _ => panic!("Expected Int32 variant"),
    }
}

#[test]
fn test_tensor_data_filled_float32() {
    let data = TensorData::filled(DataType::Float32, &[2, 3], 1.0);
    assert_eq!(data.len(), 6);

    match data {
        TensorData::Float32(ref v) => {
            assert_eq!(v.len(), 6);
            assert!(v.iter().all(|&x| x == 1.0));
        }
        _ => panic!("Expected Float32 variant"),
    }
}

#[test]
fn test_tensor_data_filled_uint8() {
    let data = TensorData::filled(DataType::Uint8, &[4, 4], 255.0);
    assert_eq!(data.len(), 16);

    match data {
        TensorData::Uint8(ref v) => {
            assert_eq!(v.len(), 16);
            assert!(v.iter().all(|&x| x == 255));
        }
        _ => panic!("Expected Uint8 variant"),
    }
}

#[test]
fn test_tensor_data_as_bytes_float32() {
    let data = TensorData::Float32(vec![1.0, 2.0, 3.0]);
    let bytes = data.as_bytes();
    assert_eq!(bytes.len(), 12); // 3 floats * 4 bytes
}

#[test]
fn test_tensor_data_as_bytes_int64() {
    let data = TensorData::Int64(vec![1, 2, 3]);
    let bytes = data.as_bytes();
    assert_eq!(bytes.len(), 24); // 3 int64s * 8 bytes
}

#[test]
fn test_tensor_data_as_bytes_uint8() {
    let data = TensorData::Uint8(vec![1, 2, 3, 4, 5]);
    let bytes = data.as_bytes();
    assert_eq!(bytes.len(), 5); // 5 uint8s * 1 byte
}

#[test]
fn test_tensor_data_len() {
    assert_eq!(TensorData::Float32(vec![1.0, 2.0, 3.0]).len(), 3);
    assert_eq!(TensorData::Int32(vec![1, 2]).len(), 2);
    assert_eq!(TensorData::Uint8(vec![1]).len(), 1);
}

#[test]
fn test_tensor_data_is_empty() {
    assert!(TensorData::Float32(vec![]).is_empty());
    assert!(!TensorData::Float32(vec![1.0]).is_empty());
}

#[test]
fn test_tensor_data_to_tensor_proto_float32() {
    let data = TensorData::Float32(vec![1.0, 2.0, 3.0]);
    let proto = data.to_tensor_proto("test".to_string(), DataType::Float32, vec![3]);

    assert_eq!(proto.name, "test");
    assert_eq!(proto.dims, vec![3]);
    assert!(!proto.raw_data.is_empty());
}

#[test]
fn test_tensor_data_to_tensor_proto_int64() {
    let data = TensorData::Int64(vec![100, 200, 300]);
    let proto = data.to_tensor_proto("weights".to_string(), DataType::Int64, vec![1, 3]);

    assert_eq!(proto.name, "weights");
    assert_eq!(proto.dims, vec![1, 3]);
    assert!(!proto.raw_data.is_empty());
}

#[test]
fn test_tensor_data_all_types() {
    let types = [
        (DataType::Float32, 1.0),
        (DataType::Int32, 2.0),
        (DataType::Int64, 3.0),
        (DataType::Int8, 4.0),
        (DataType::Uint32, 5.0),
        (DataType::Uint64, 6.0),
        (DataType::Uint8, 7.0),
    ];

    for (dtype, value) in types {
        let data = TensorData::scalar(dtype, value);
        assert_eq!(data.len(), 1);
        assert!(!data.is_empty());
    }
}
