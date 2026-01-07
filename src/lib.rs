//! Shared utilities for ONNX <-> WebNN conversion.

pub mod protos;

pub mod attributes;
pub mod data_types;
pub mod error;
pub mod identifiers;
pub mod operation_names;
pub mod shape_inference;
pub mod tensor_data;

pub use error::{ConversionError, Result};
