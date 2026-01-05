//! Shared utilities for ONNX <-> WebNN conversion.

pub mod protos;

pub mod error;
pub mod data_types;
pub mod operation_names;
pub mod attributes;
pub mod tensor_data;
pub mod identifiers;
pub mod shape_inference;

pub use error::{ConversionError, Result};
