use serde_json::Value as JsonValue;

use crate::error::{ConversionError, Result};
use crate::protos::onnx::{attribute_proto, AttributeProto};

#[derive(Debug, Clone)]
pub enum AttrValue {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
}

pub struct AttrParser<'a> {
    attrs: &'a [AttributeProto],
}

impl<'a> AttrParser<'a> {
    pub fn new(attrs: &'a [AttributeProto]) -> Self {
        Self { attrs }
    }

    pub fn get_int(&self, name: &str) -> Option<i64> {
        let a = self.attrs.iter().find(|a| a.name == name)?;
        if a.r#type == attribute_proto::AttributeType::Int as i32 {
            Some(a.i)
        } else {
            None
        }
    }

    pub fn get_ints(&self, name: &str) -> Option<Vec<i64>> {
        let a = self.attrs.iter().find(|a| a.name == name)?;
        if a.r#type == attribute_proto::AttributeType::Ints as i32 && !a.ints.is_empty() {
            Some(a.ints.clone())
        } else {
            None
        }
    }
}

#[derive(Default)]
pub struct AttrBuilder {
    attrs: Vec<AttributeProto>,
}

impl AttrBuilder {
    pub fn new() -> Self {
        Self { attrs: vec![] }
    }

    pub fn add_int(&mut self, name: &str, value: i64) -> &mut Self {
        let mut a = AttributeProto::default();
        a.name = name.to_string();
        a.r#type = attribute_proto::AttributeType::Int as i32;
        a.i = value;
        self.attrs.push(a);
        self
    }

    pub fn add_ints(&mut self, name: &str, values: Vec<i64>) -> &mut Self {
        let mut a = AttributeProto::default();
        a.name = name.to_string();
        a.ints = values;
        self.attrs.push(a);
        self
    }

    pub fn build(self) -> Vec<AttributeProto> {
        self.attrs
    }
}

pub fn parse_json_ints(json: &JsonValue, key: &str) -> Option<Vec<i64>> {
    json.get(key)?
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<_>>())
}

pub fn require_attr<T>(name: &str, v: Option<T>) -> Result<T> {
    v.ok_or_else(|| ConversionError::InvalidAttribute(format!("missing attribute: {name}")))
}
