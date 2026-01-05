use thiserror::Error;

pub type Result<T> = std::result::Result<T, ConversionError>;

#[derive(Debug, Error)]
pub enum ConversionError {
    #[error("unsupported data type code: {0}")]
    UnsupportedOnnxDataType(i32),

    #[error("invalid attribute: {0}")]
    InvalidAttribute(String),

    #[error("invalid tensor data: {0}")]
    InvalidTensorData(String),

    #[error("internal error: {0}")]
    Internal(String),
}
