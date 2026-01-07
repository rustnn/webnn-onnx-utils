pub fn sanitize_for_webnn(onnx_name: &str) -> String {
    onnx_name.replace("::", "__").replace([':', '.', '/'], "_")
}

pub fn is_valid_webnn_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

pub fn make_unique(base: &str, counter: &mut usize) -> String {
    let name = format!("{base}_{counter}");
    *counter += 1;
    name
}
