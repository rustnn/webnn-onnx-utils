use crate::data_types::{DataType, onnx_proto_to_webnn};
use crate::error::{ConversionError, Result};
use crate::protos::onnx::TensorProto;

#[derive(Debug, Clone)]
pub enum TensorData {
    Raw(Vec<u8>),
    Float32(Vec<f32>),
    Float16(Vec<u16>), // Raw bits representation
    Int64(Vec<i64>),
    Int32(Vec<i32>),
    Int8(Vec<i8>),
    Uint64(Vec<u64>),
    Uint32(Vec<u32>),
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
            DataType::Float32 => {
                if !tensor.float_data.is_empty() {
                    Ok(TensorData::Float32(tensor.float_data.clone()))
                } else {
                    Ok(TensorData::Float32(vec![]))
                }
            }
            DataType::Float16 => {
                // Float16 data stored as int32_data in ONNX
                if !tensor.int32_data.is_empty() {
                    Ok(TensorData::Float16(
                        tensor.int32_data.iter().map(|v| *v as u16).collect(),
                    ))
                } else {
                    Ok(TensorData::Float16(vec![]))
                }
            }
            DataType::Int64 => {
                if !tensor.int64_data.is_empty() {
                    Ok(TensorData::Int64(tensor.int64_data.clone()))
                } else {
                    Ok(TensorData::Int64(vec![]))
                }
            }
            DataType::Int32 => {
                if !tensor.int32_data.is_empty() {
                    Ok(TensorData::Int32(tensor.int32_data.to_vec()))
                } else {
                    Ok(TensorData::Int32(vec![]))
                }
            }
            DataType::Int8 => {
                if !tensor.int32_data.is_empty() {
                    Ok(TensorData::Int8(
                        tensor.int32_data.iter().map(|v| *v as i8).collect(),
                    ))
                } else {
                    Ok(TensorData::Int8(vec![]))
                }
            }
            DataType::Uint64 => {
                if !tensor.uint64_data.is_empty() {
                    Ok(TensorData::Uint64(tensor.uint64_data.clone()))
                } else {
                    Ok(TensorData::Uint64(vec![]))
                }
            }
            DataType::Uint32 => {
                if !tensor.uint64_data.is_empty() {
                    Ok(TensorData::Uint32(
                        tensor.uint64_data.iter().map(|v| *v as u32).collect(),
                    ))
                } else {
                    Ok(TensorData::Uint32(vec![]))
                }
            }
            DataType::Uint8 => {
                if !tensor.int32_data.is_empty() {
                    Ok(TensorData::Uint8(
                        tensor.int32_data.iter().map(|v| *v as u8).collect(),
                    ))
                } else {
                    Ok(TensorData::Uint8(vec![]))
                }
            }
        }
    }

    pub fn to_tensor_proto(&self, name: String, dtype: DataType, shape: Vec<i64>) -> TensorProto {
        TensorProto {
            name,
            data_type: crate::data_types::webnn_to_onnx(dtype) as i32,
            dims: shape,
            raw_data: self.as_bytes().to_vec(),
            ..Default::default()
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            TensorData::Raw(v) => v,
            TensorData::Uint8(v) => v,
            TensorData::Float32(v) => unsafe {
                std::slice::from_raw_parts(
                    v.as_ptr() as *const u8,
                    v.len() * std::mem::size_of::<f32>(),
                )
            },
            TensorData::Float16(v) => unsafe {
                std::slice::from_raw_parts(
                    v.as_ptr() as *const u8,
                    v.len() * std::mem::size_of::<u16>(),
                )
            },
            TensorData::Int64(v) => unsafe {
                std::slice::from_raw_parts(
                    v.as_ptr() as *const u8,
                    v.len() * std::mem::size_of::<i64>(),
                )
            },
            TensorData::Int32(v) => unsafe {
                std::slice::from_raw_parts(
                    v.as_ptr() as *const u8,
                    v.len() * std::mem::size_of::<i32>(),
                )
            },
            TensorData::Int8(v) => unsafe {
                std::slice::from_raw_parts(
                    v.as_ptr() as *const u8,
                    v.len() * std::mem::size_of::<i8>(),
                )
            },
            TensorData::Uint64(v) => unsafe {
                std::slice::from_raw_parts(
                    v.as_ptr() as *const u8,
                    v.len() * std::mem::size_of::<u64>(),
                )
            },
            TensorData::Uint32(v) => unsafe {
                std::slice::from_raw_parts(
                    v.as_ptr() as *const u8,
                    v.len() * std::mem::size_of::<u32>(),
                )
            },
        }
    }

    pub fn len(&self) -> usize {
        match self {
            TensorData::Raw(v) => v.len(),
            TensorData::Float32(v) => v.len(),
            TensorData::Float16(v) => v.len(),
            TensorData::Int64(v) => v.len(),
            TensorData::Int32(v) => v.len(),
            TensorData::Int8(v) => v.len(),
            TensorData::Uint64(v) => v.len(),
            TensorData::Uint32(v) => v.len(),
            TensorData::Uint8(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn scalar(dtype: DataType, value: f32) -> Self {
        match dtype {
            DataType::Float32 => TensorData::Float32(vec![value]),
            DataType::Float16 => TensorData::Float16(vec![half::f16::from_f32(value).to_bits()]),
            DataType::Int32 => TensorData::Int32(vec![value as i32]),
            DataType::Int64 => TensorData::Int64(vec![value as i64]),
            DataType::Int8 => TensorData::Int8(vec![value as i8]),
            DataType::Uint32 => TensorData::Uint32(vec![value as u32]),
            DataType::Uint64 => TensorData::Uint64(vec![value as u64]),
            DataType::Uint8 => TensorData::Uint8(vec![value as u8]),
        }
    }

    pub fn filled(dtype: DataType, shape: &[i64], value: f32) -> Self {
        let count = shape.iter().product::<i64>() as usize;
        match dtype {
            DataType::Float32 => TensorData::Float32(vec![value; count]),
            DataType::Float16 => {
                let bits = half::f16::from_f32(value).to_bits();
                TensorData::Float16(vec![bits; count])
            }
            DataType::Int32 => TensorData::Int32(vec![value as i32; count]),
            DataType::Int64 => TensorData::Int64(vec![value as i64; count]),
            DataType::Int8 => TensorData::Int8(vec![value as i8; count]),
            DataType::Uint32 => TensorData::Uint32(vec![value as u32; count]),
            DataType::Uint64 => TensorData::Uint64(vec![value as u64; count]),
            DataType::Uint8 => TensorData::Uint8(vec![value as u8; count]),
        }
    }
}
