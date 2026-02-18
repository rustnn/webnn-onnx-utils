use crate::protos::onnx::TensorShapeProto;
use crate::protos::onnx::tensor_shape_proto::{
    Dimension as OnnxDimension, dimension::Value as DimValue,
};
use crate::shape_inference::Dim;

/// Convert a shape inference Dim into an ONNX Dimension.
pub fn dim_to_onnx(dim: &Dim) -> OnnxDimension {
    match dim {
        Dim::Known(v) => OnnxDimension {
            value: Some(DimValue::DimValue(*v)),
            denotation: String::new(),
        },
        Dim::Dynamic(name) => OnnxDimension {
            value: if name.is_empty() {
                None
            } else {
                Some(DimValue::DimParam(name.clone()))
            },
            denotation: String::new(),
        },
    }
}

/// Convert a list of Dim into ONNX Dimensions.
pub fn dims_to_onnx(dims: &[Dim]) -> Vec<OnnxDimension> {
    dims.iter().map(dim_to_onnx).collect()
}

/// Build an ONNX TensorShapeProto from Dim list.
pub fn to_onnx_shape_proto(dims: &[Dim]) -> TensorShapeProto {
    TensorShapeProto {
        dim: dims_to_onnx(dims),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dim_to_onnx_known() {
        let dim = Dim::Known(7);
        let onnx = dim_to_onnx(&dim);
        match onnx.value {
            Some(DimValue::DimValue(v)) => assert_eq!(v, 7),
            _ => panic!("expected DimValue"),
        }
    }

    #[test]
    fn test_dim_to_onnx_dynamic() {
        let dim = Dim::Dynamic("batch".to_string());
        let onnx = dim_to_onnx(&dim);
        match onnx.value {
            Some(DimValue::DimParam(p)) => assert_eq!(p, "batch"),
            _ => panic!("expected DimParam"),
        }
    }

    #[test]
    fn test_dim_to_onnx_empty_param() {
        let dim = Dim::Dynamic(String::new());
        let onnx = dim_to_onnx(&dim);
        assert!(onnx.value.is_none());
    }
}
