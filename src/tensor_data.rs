use crate::data_types::{onnx_proto_to_webnn, DataType};
use crate::error::{ConversionError, Result};
use crate::protos::onnx::TensorProto;

#[derive(Debug, Clone)]
pub enum TensorData {
    Raw(Vec<u8>),
    Float32(Vec<f32>),
    Int64(Vec<i64>),
    Int32(Vec<i32>),
    Uint8(Vec<u8>),
}

impl TensorData {
    pub fn from_tensor_proto(tensor: &TensorProto) -> Result<Self> {
        let proto = crate::protos::onnx::tensor_proto::DataType::try_from(tensor.data_type)
            .map_err(|_| ConversionError::UnsupportedOnnxDataType(tensor.data_type))?;
        let dtype = onnx_proto_to_webnn(proto)?;

        if !tensor.raw_data.is_empty() {
            return Ok(TensorData::Raw(tensor.raw_data.clone()));
        }

        match dtype {
            DataType::Float32 => Ok(TensorData::Float32(tensor.float_data.clone())),
            DataType::Int64 => Ok(TensorData::Int64(tensor.int64_data.clone())),
            DataType::Int32 => Ok(TensorData::Int32(
                tensor.int32_data.iter().map(|v| *v as i32).collect(),
            )),
            DataType::Uint8 => Ok(TensorData::Uint8(
                tensor.int32_data.iter().map(|v| *v as u8).collect(),
            )),
            other => Err(ConversionError::InvalidTensorData(format!(
                "typed field decoding not implemented for {other:?} without raw_data"
            ))),
        }
    }

    pub fn to_tensor_proto(&self, name: String, dtype: DataType, shape: Vec<i64>) -> TensorProto {
        let mut t = TensorProto::default();
        t.name = name;
        t.data_type = crate::data_types::webnn_to_onnx(dtype) as i32;
        t.dims = shape;
        t.raw_data = self.as_bytes().to_vec();
        t
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            TensorData::Raw(v) => v,
            TensorData::Uint8(v) => v,
            _ => &[],
        }
    }
}
