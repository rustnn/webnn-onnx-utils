use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum Dim {
    Known(i64),
    Dynamic(String),
}

#[derive(Debug, Clone)]
pub struct TensorShape {
    pub dims: Vec<Dim>,
}

impl TensorShape {
    pub fn is_fully_static(&self) -> bool {
        self.dims.iter().all(|d| matches!(d, Dim::Known(_)))
    }

    pub fn to_static(&self, overrides: &HashMap<String, u32>) -> Option<Vec<i64>> {
        self.dims
            .iter()
            .map(|d| match d {
                Dim::Known(v) => Some(*v),
                Dim::Dynamic(k) => overrides.get(k).map(|v| *v as i64),
            })
            .collect()
    }
}
