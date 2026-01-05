use serde::{Deserialize, Serialize};

use crate::error::{ConversionError, Result};
use crate::protos::onnx::tensor_proto::DataType as ProtoDataType;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    Float32,
    Float16,
    Int32,
    Uint32,
    Int64,
    Uint64,
    Int8,
    Uint8,
}

pub fn webnn_to_onnx(webnn_type: DataType) -> ProtoDataType {
    match webnn_type {
        DataType::Float32 => ProtoDataType::Float,
        DataType::Float16 => ProtoDataType::Float16,
        DataType::Int32 => ProtoDataType::Int32,
        DataType::Uint32 => ProtoDataType::Uint32,
        DataType::Int64 => ProtoDataType::Int64,
        DataType::Uint64 => ProtoDataType::Uint64,
        DataType::Int8 => ProtoDataType::Int8,
        DataType::Uint8 => ProtoDataType::Uint8,
    }
}

pub fn onnx_to_webnn(onnx_type_code: i32) -> Result<DataType> {
    match ProtoDataType::try_from(onnx_type_code) {
        Ok(proto) => onnx_proto_to_webnn(proto),
        Err(_) => Err(ConversionError::UnsupportedOnnxDataType(onnx_type_code)),
    }
}

pub fn onnx_proto_to_webnn(proto_type: ProtoDataType) -> Result<DataType> {
    match proto_type {
        ProtoDataType::Float => Ok(DataType::Float32),
        ProtoDataType::Float16 => Ok(DataType::Float16),
        ProtoDataType::Int32 => Ok(DataType::Int32),
        ProtoDataType::Uint32 => Ok(DataType::Uint32),
        ProtoDataType::Int64 => Ok(DataType::Int64),
        ProtoDataType::Uint64 => Ok(DataType::Uint64),
        ProtoDataType::Int8 => Ok(DataType::Int8),
        ProtoDataType::Uint8 => Ok(DataType::Uint8),
        other => Err(ConversionError::UnsupportedOnnxDataType(other as i32)),
    }
}
