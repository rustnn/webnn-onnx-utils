use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    Known(i64),
    Dynamic(String),
}

#[derive(Debug, Clone)]
pub struct TensorShape {
    pub dims: Vec<Dim>,
}

impl TensorShape {
    pub fn new(dims: Vec<Dim>) -> Self {
        Self { dims }
    }

    pub fn from_static(dims: Vec<i64>) -> Self {
        Self {
            dims: dims.into_iter().map(Dim::Known).collect(),
        }
    }

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

    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

pub struct ShapeInferenceContext {
    value_shapes: HashMap<String, TensorShape>,
    overrides: HashMap<String, u32>,
}

impl ShapeInferenceContext {
    pub fn new() -> Self {
        Self {
            value_shapes: HashMap::new(),
            overrides: HashMap::new(),
        }
    }

    pub fn with_overrides(overrides: HashMap<String, u32>) -> Self {
        Self {
            value_shapes: HashMap::new(),
            overrides,
        }
    }

    pub fn set_shape(&mut self, name: String, shape: TensorShape) {
        self.value_shapes.insert(name, shape);
    }

    pub fn get_shape(&self, name: &str) -> Option<&TensorShape> {
        self.value_shapes.get(name)
    }

    pub fn infer_unary_op(&self, input: &str) -> Option<TensorShape> {
        self.get_shape(input).cloned()
    }

    pub fn infer_binary_op(&self, a: &str, b: &str) -> Option<TensorShape> {
        let shape_a = self.get_shape(a)?;
        let shape_b = self.get_shape(b)?;

        let static_a = shape_a.to_static(&self.overrides)?;
        let static_b = shape_b.to_static(&self.overrides)?;

        let result = broadcast_shapes(&static_a, &static_b)?;
        Some(TensorShape::from_static(result))
    }

    pub fn infer_matmul(&self, a: &str, b: &str) -> Option<TensorShape> {
        let shape_a = self.get_shape(a)?;
        let shape_b = self.get_shape(b)?;

        let static_a = shape_a.to_static(&self.overrides)?;
        let static_b = shape_b.to_static(&self.overrides)?;

        if static_a.len() < 2 || static_b.len() < 2 {
            return None;
        }

        let m = static_a[static_a.len() - 2];
        let k_a = static_a[static_a.len() - 1];
        let k_b = static_b[static_b.len() - 2];
        let n = static_b[static_b.len() - 1];

        if k_a != k_b {
            return None;
        }

        let mut result = Vec::new();

        // Handle batch dimensions
        if static_a.len() > 2 || static_b.len() > 2 {
            let batch_a = &static_a[..static_a.len() - 2];
            let batch_b = &static_b[..static_b.len() - 2];
            let batch = broadcast_shapes(batch_a, batch_b)?;
            result.extend(batch);
        }

        result.push(m);
        result.push(n);

        Some(TensorShape::from_static(result))
    }

    pub fn infer_transpose(&self, input: &str, perm: &[usize]) -> Option<TensorShape> {
        let shape = self.get_shape(input)?;

        if perm.len() != shape.dims.len() {
            return None;
        }

        let mut result_dims = vec![Dim::Known(0); perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            if p >= shape.dims.len() {
                return None;
            }
            result_dims[i] = shape.dims[p].clone();
        }

        Some(TensorShape::new(result_dims))
    }

    pub fn infer_reduce(&self, input: &str, axes: &[i64], keep_dims: bool) -> Option<TensorShape> {
        let shape = self.get_shape(input)?;
        let static_shape = shape.to_static(&self.overrides)?;

        let mut result = Vec::new();
        for (i, &dim) in static_shape.iter().enumerate() {
            let i_signed = i as i64;
            if axes.contains(&i_signed) || axes.contains(&(i_signed - static_shape.len() as i64)) {
                if keep_dims {
                    result.push(1);
                }
            } else {
                result.push(dim);
            }
        }

        Some(TensorShape::from_static(result))
    }

    pub fn infer_concat(&self, inputs: &[&str], axis: i64) -> Option<TensorShape> {
        if inputs.is_empty() {
            return None;
        }

        let first_shape = self.get_shape(inputs[0])?;
        let static_first = first_shape.to_static(&self.overrides)?;

        let rank = static_first.len() as i64;
        let normalized_axis = if axis < 0 { axis + rank } else { axis };

        if normalized_axis < 0 || normalized_axis >= rank {
            return None;
        }

        let mut result = static_first.clone();
        let axis_idx = normalized_axis as usize;

        for &input in &inputs[1..] {
            let shape = self.get_shape(input)?;
            let static_shape = shape.to_static(&self.overrides)?;

            if static_shape.len() != result.len() {
                return None;
            }

            for (i, &dim_b) in static_shape.iter().enumerate() {
                if i == axis_idx {
                    result[i] += dim_b;
                } else if result[i] != dim_b {
                    return None;
                }
            }
        }

        Some(TensorShape::from_static(result))
    }

    pub fn infer_reshape(&self, input: &str, new_shape: &[i64]) -> Option<TensorShape> {
        let shape = self.get_shape(input)?;
        let static_shape = shape.to_static(&self.overrides)?;

        let total_elements: i64 = static_shape.iter().product();

        let neg_one_count = new_shape.iter().filter(|&&x| x == -1).count();
        if neg_one_count > 1 {
            return None;
        }

        let mut result = new_shape.to_vec();

        if neg_one_count == 1 {
            let known_product: i64 = new_shape.iter().filter(|&&x| x > 0).product();
            let inferred = total_elements / known_product;

            for dim in &mut result {
                if *dim == -1 {
                    *dim = inferred;
                    break;
                }
            }
        }

        Some(TensorShape::from_static(result))
    }

    pub fn infer_squeeze(&self, input: &str, axes: &[i64]) -> Option<TensorShape> {
        let shape = self.get_shape(input)?;
        let static_shape = shape.to_static(&self.overrides)?;

        let mut result = Vec::new();

        if axes.is_empty() {
            for &dim in &static_shape {
                if dim != 1 {
                    result.push(dim);
                }
            }
        } else {
            for (i, &dim) in static_shape.iter().enumerate() {
                let i_signed = i as i64;
                let should_squeeze = axes.contains(&i_signed)
                    || axes.contains(&(i_signed - static_shape.len() as i64));

                if !should_squeeze {
                    result.push(dim);
                } else if dim != 1 {
                    return None;
                }
            }
        }

        Some(TensorShape::from_static(result))
    }

    pub fn infer_unsqueeze(&self, input: &str, axes: &[i64]) -> Option<TensorShape> {
        let shape = self.get_shape(input)?;
        let static_shape = shape.to_static(&self.overrides)?;

        let new_rank = static_shape.len() + axes.len();
        let mut result = vec![0i64; new_rank];

        let mut normalized_axes: Vec<usize> = axes
            .iter()
            .map(|&ax| {
                let normalized = if ax < 0 { ax + new_rank as i64 } else { ax };
                normalized as usize
            })
            .collect();
        normalized_axes.sort_unstable();

        let mut input_idx = 0;
        for (i, item) in result.iter_mut().enumerate() {
            if normalized_axes.contains(&i) {
                *item = 1;
            } else {
                *item = static_shape[input_idx];
                input_idx += 1;
            }
        }

        Some(TensorShape::from_static(result))
    }
}

impl Default for ShapeInferenceContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Broadcast shapes following NumPy broadcasting rules
pub fn broadcast_shapes(a: &[i64], b: &[i64]) -> Option<Vec<i64>> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let dim_a = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let dim_b = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if dim_a == dim_b {
            result.push(dim_a);
        } else if dim_a == 1 {
            result.push(dim_b);
        } else if dim_b == 1 {
            result.push(dim_a);
        } else {
            return None;
        }
    }

    result.reverse();
    Some(result)
}
