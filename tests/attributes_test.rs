use webnn_onnx_utils::attributes::{AttrBuilder, AttrParser};

#[test]
fn test_attr_builder_int() {
    let attrs = AttrBuilder::new().add_int("test", 42).build();

    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name, "test");
    assert_eq!(attrs[0].i, 42);
}

#[test]
fn test_attr_builder_ints() {
    let attrs = AttrBuilder::new()
        .add_ints("strides", vec![1, 2, 3])
        .build();

    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name, "strides");
    assert_eq!(attrs[0].ints, vec![1, 2, 3]);
}

#[test]
fn test_attr_builder_float() {
    let attrs = AttrBuilder::new().add_float("alpha", 0.5).build();

    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name, "alpha");
    assert_eq!(attrs[0].f, 0.5);
}

#[test]
fn test_attr_builder_floats() {
    let attrs = AttrBuilder::new()
        .add_floats("scales", vec![1.0, 2.0, 3.0])
        .build();

    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name, "scales");
    assert_eq!(attrs[0].floats, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_attr_builder_string() {
    let attrs = AttrBuilder::new()
        .add_string("mode", "linear".to_string())
        .build();

    assert_eq!(attrs.len(), 1);
    assert_eq!(attrs[0].name, "mode");
    assert_eq!(String::from_utf8_lossy(&attrs[0].s), "linear");
}

#[test]
fn test_attr_builder_multiple() {
    let attrs = AttrBuilder::new()
        .add_int("axis", 1)
        .add_float("epsilon", 1e-5)
        .add_ints("perm", vec![0, 2, 1, 3])
        .build();

    assert_eq!(attrs.len(), 3);
}

#[test]
fn test_attr_parser_int() {
    let attrs = AttrBuilder::new().add_int("axis", 2).build();
    let parser = AttrParser::new(&attrs);

    assert_eq!(parser.get_int("axis"), Some(2));
    assert_eq!(parser.get_int("missing"), None);
}

#[test]
fn test_attr_parser_ints() {
    let attrs = AttrBuilder::new().add_ints("dilations", vec![1, 1]).build();
    let parser = AttrParser::new(&attrs);

    assert_eq!(parser.get_ints("dilations"), Some(vec![1, 1]));
    assert_eq!(parser.get_ints("missing"), None);
}

#[test]
fn test_attr_parser_float() {
    let attrs = AttrBuilder::new().add_float("momentum", 0.9).build();
    let parser = AttrParser::new(&attrs);

    assert_eq!(parser.get_float("momentum"), Some(0.9));
    assert_eq!(parser.get_float("missing"), None);
}

#[test]
fn test_attr_parser_floats() {
    let attrs = AttrBuilder::new()
        .add_floats("coefficients", vec![0.1, 0.2, 0.3])
        .build();
    let parser = AttrParser::new(&attrs);

    assert_eq!(parser.get_floats("coefficients"), Some(vec![0.1, 0.2, 0.3]));
    assert_eq!(parser.get_floats("missing"), None);
}

#[test]
fn test_attr_parser_string() {
    let attrs = AttrBuilder::new()
        .add_string("activation", "relu".to_string())
        .build();
    let parser = AttrParser::new(&attrs);

    assert_eq!(parser.get_string("activation"), Some("relu".to_string()));
    assert_eq!(parser.get_string("missing"), None);
}
