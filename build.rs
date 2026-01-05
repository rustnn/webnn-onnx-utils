use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=protos/onnx/onnx.proto3");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR must be set"));

    prost_build::Config::new()
        .out_dir(out_dir)
        .compile_protos(&["protos/onnx/onnx.proto3"], &["protos/onnx"])
        .expect("failed to compile ONNX protos with prost-build");
}
