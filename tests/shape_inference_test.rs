use webnn_onnx_utils::shape_inference::{ShapeInferenceContext, TensorShape, broadcast_shapes};

#[test]
fn test_broadcast_shapes_same_rank() {
    let a = vec![3, 4, 5];
    let b = vec![3, 4, 5];
    let result = broadcast_shapes(&a, &b);
    assert_eq!(result, Some(vec![3, 4, 5]));
}

#[test]
fn test_broadcast_shapes_with_ones() {
    let a = vec![3, 1, 5];
    let b = vec![3, 4, 1];
    let result = broadcast_shapes(&a, &b);
    assert_eq!(result, Some(vec![3, 4, 5]));
}

#[test]
fn test_broadcast_shapes_different_ranks() {
    let a = vec![5];
    let b = vec![3, 4, 5];
    let result = broadcast_shapes(&a, &b);
    assert_eq!(result, Some(vec![3, 4, 5]));
}

#[test]
fn test_broadcast_shapes_incompatible() {
    let a = vec![3, 4];
    let b = vec![3, 5];
    let result = broadcast_shapes(&a, &b);
    assert_eq!(result, None);
}

#[test]
fn test_infer_unary_op() {
    let mut ctx = ShapeInferenceContext::new();
    ctx.set_shape("input".to_string(), TensorShape::from_static(vec![2, 3, 4]));

    let result = ctx.infer_unary_op("input");
    assert!(result.is_some());
    let shape = result.unwrap();
    assert_eq!(shape.dims.len(), 3);
}

#[test]
fn test_infer_binary_op() {
    let mut ctx = ShapeInferenceContext::new();
    ctx.set_shape("a".to_string(), TensorShape::from_static(vec![3, 1, 5]));
    ctx.set_shape("b".to_string(), TensorShape::from_static(vec![3, 4, 1]));

    let result = ctx.infer_binary_op("a", "b");
    assert!(result.is_some());
    let shape = result.unwrap();
    assert_eq!(shape.to_static(&Default::default()), Some(vec![3, 4, 5]));
}

#[test]
fn test_infer_matmul_2d() {
    let mut ctx = ShapeInferenceContext::new();
    ctx.set_shape("a".to_string(), TensorShape::from_static(vec![3, 4]));
    ctx.set_shape("b".to_string(), TensorShape::from_static(vec![4, 5]));

    let result = ctx.infer_matmul("a", "b");
    assert!(result.is_some());
    let shape = result.unwrap();
    assert_eq!(shape.to_static(&Default::default()), Some(vec![3, 5]));
}

#[test]
fn test_infer_matmul_batched() {
    let mut ctx = ShapeInferenceContext::new();
    ctx.set_shape("a".to_string(), TensorShape::from_static(vec![2, 3, 4]));
    ctx.set_shape("b".to_string(), TensorShape::from_static(vec![2, 4, 5]));

    let result = ctx.infer_matmul("a", "b");
    assert!(result.is_some());
    let shape = result.unwrap();
    assert_eq!(shape.to_static(&Default::default()), Some(vec![2, 3, 5]));
}

#[test]
fn test_infer_matmul_incompatible() {
    let mut ctx = ShapeInferenceContext::new();
    ctx.set_shape("a".to_string(), TensorShape::from_static(vec![3, 4]));
    ctx.set_shape("b".to_string(), TensorShape::from_static(vec![5, 6]));

    let result = ctx.infer_matmul("a", "b");
    assert!(result.is_none());
}

#[test]
fn test_infer_transpose() {
    let mut ctx = ShapeInferenceContext::new();
    ctx.set_shape("input".to_string(), TensorShape::from_static(vec![2, 3, 4]));

    let result = ctx.infer_transpose("input", &[0, 2, 1]);
    assert!(result.is_some());
    let shape = result.unwrap();
    assert_eq!(shape.to_static(&Default::default()), Some(vec![2, 4, 3]));
}

#[test]
fn test_infer_reduce_keep_dims() {
    let mut ctx = ShapeInferenceContext::new();
    ctx.set_shape("input".to_string(), TensorShape::from_static(vec![2, 3, 4]));

    let result = ctx.infer_reduce("input", &[1], true);
    assert!(result.is_some());
    let shape = result.unwrap();
    assert_eq!(shape.to_static(&Default::default()), Some(vec![2, 1, 4]));
}

#[test]
fn test_infer_reduce_no_keep_dims() {
    let mut ctx = ShapeInferenceContext::new();
    ctx.set_shape("input".to_string(), TensorShape::from_static(vec![2, 3, 4]));

    let result = ctx.infer_reduce("input", &[1], false);
    assert!(result.is_some());
    let shape = result.unwrap();
    assert_eq!(shape.to_static(&Default::default()), Some(vec![2, 4]));
}

#[test]
fn test_infer_concat() {
    let mut ctx = ShapeInferenceContext::new();
    ctx.set_shape("a".to_string(), TensorShape::from_static(vec![2, 3, 4]));
    ctx.set_shape("b".to_string(), TensorShape::from_static(vec![2, 5, 4]));
    ctx.set_shape("c".to_string(), TensorShape::from_static(vec![2, 7, 4]));

    let result = ctx.infer_concat(&["a", "b", "c"], 1);
    assert!(result.is_some());
    let shape = result.unwrap();
    assert_eq!(shape.to_static(&Default::default()), Some(vec![2, 15, 4]));
}

#[test]
fn test_infer_reshape() {
    let mut ctx = ShapeInferenceContext::new();
    ctx.set_shape("input".to_string(), TensorShape::from_static(vec![2, 3, 4]));

    let result = ctx.infer_reshape("input", &[6, 4]);
    assert!(result.is_some());
    let shape = result.unwrap();
    assert_eq!(shape.to_static(&Default::default()), Some(vec![6, 4]));
}

#[test]
fn test_infer_reshape_with_negative_one() {
    let mut ctx = ShapeInferenceContext::new();
    ctx.set_shape("input".to_string(), TensorShape::from_static(vec![2, 3, 4]));

    let result = ctx.infer_reshape("input", &[-1, 4]);
    assert!(result.is_some());
    let shape = result.unwrap();
    assert_eq!(shape.to_static(&Default::default()), Some(vec![6, 4]));
}

#[test]
fn test_infer_squeeze_all() {
    let mut ctx = ShapeInferenceContext::new();
    ctx.set_shape(
        "input".to_string(),
        TensorShape::from_static(vec![1, 3, 1, 4]),
    );

    let result = ctx.infer_squeeze("input", &[]);
    assert!(result.is_some());
    let shape = result.unwrap();
    assert_eq!(shape.to_static(&Default::default()), Some(vec![3, 4]));
}

#[test]
fn test_infer_squeeze_specific_axes() {
    let mut ctx = ShapeInferenceContext::new();
    ctx.set_shape(
        "input".to_string(),
        TensorShape::from_static(vec![1, 3, 1, 4]),
    );

    let result = ctx.infer_squeeze("input", &[0, 2]);
    assert!(result.is_some());
    let shape = result.unwrap();
    assert_eq!(shape.to_static(&Default::default()), Some(vec![3, 4]));
}

#[test]
fn test_infer_unsqueeze() {
    let mut ctx = ShapeInferenceContext::new();
    ctx.set_shape("input".to_string(), TensorShape::from_static(vec![3, 4]));

    let result = ctx.infer_unsqueeze("input", &[0, 2]);
    assert!(result.is_some());
    let shape = result.unwrap();
    assert_eq!(shape.to_static(&Default::default()), Some(vec![1, 3, 1, 4]));
}
