use serde_json::Value as JsonValue;

use crate::error::{ConversionError, Result};
use crate::protos::onnx::{AttributeProto, attribute_proto};

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

    pub fn get_float(&self, name: &str) -> Option<f32> {
        let a = self.attrs.iter().find(|a| a.name == name)?;
        if a.r#type == attribute_proto::AttributeType::Float as i32 {
            Some(a.f)
        } else {
            None
        }
    }

    pub fn get_floats(&self, name: &str) -> Option<Vec<f32>> {
        let a = self.attrs.iter().find(|a| a.name == name)?;
        if a.r#type == attribute_proto::AttributeType::Floats as i32 && !a.floats.is_empty() {
            Some(a.floats.clone())
        } else {
            None
        }
    }

    pub fn get_string(&self, name: &str) -> Option<String> {
        let a = self.attrs.iter().find(|a| a.name == name)?;
        if a.r#type == attribute_proto::AttributeType::String as i32 {
            Some(String::from_utf8_lossy(&a.s).to_string())
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

    pub fn add_int(mut self, name: &str, value: i64) -> Self {
        let a = AttributeProto {
            name: name.to_string(),
            r#type: attribute_proto::AttributeType::Int as i32,
            i: value,
            ..Default::default()
        };
        self.attrs.push(a);
        self
    }

    pub fn add_ints(mut self, name: &str, values: Vec<i64>) -> Self {
        let a = AttributeProto {
            name: name.to_string(),
            r#type: attribute_proto::AttributeType::Ints as i32,
            ints: values,
            ..Default::default()
        };
        self.attrs.push(a);
        self
    }

    pub fn add_float(mut self, name: &str, value: f32) -> Self {
        let a = AttributeProto {
            name: name.to_string(),
            r#type: attribute_proto::AttributeType::Float as i32,
            f: value,
            ..Default::default()
        };
        self.attrs.push(a);
        self
    }

    pub fn add_floats(mut self, name: &str, values: Vec<f32>) -> Self {
        let a = AttributeProto {
            name: name.to_string(),
            r#type: attribute_proto::AttributeType::Floats as i32,
            floats: values,
            ..Default::default()
        };
        self.attrs.push(a);
        self
    }

    pub fn add_string(mut self, name: &str, value: String) -> Self {
        let a = AttributeProto {
            name: name.to_string(),
            r#type: attribute_proto::AttributeType::String as i32,
            s: value.into_bytes(),
            ..Default::default()
        };
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

pub fn parse_json_floats(json: &JsonValue, key: &str) -> Option<Vec<f32>> {
    json.get(key)?.as_array().map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect::<Vec<_>>()
    })
}

pub fn require_attr<T>(name: &str, v: Option<T>) -> Result<T> {
    v.ok_or_else(|| ConversionError::InvalidAttribute(format!("missing attribute: {name}")))
}
