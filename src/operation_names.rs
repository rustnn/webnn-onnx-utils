use std::collections::HashMap;

use once_cell::sync::Lazy;

#[derive(Debug, Clone)]
pub struct OpNameMapper {
    webnn_to_onnx: HashMap<String, String>,
    onnx_to_webnn: HashMap<String, String>,
}

impl OpNameMapper {
    pub fn new() -> Self {
        let mut webnn_to_onnx = HashMap::new();
        let mut onnx_to_webnn = HashMap::new();

        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "matmul", "MatMul");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "conv2d", "Conv");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "convTranspose2d", "ConvTranspose");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "averagePool2d", "AveragePool");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "maxPool2d", "MaxPool");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "globalAveragePool", "GlobalAveragePool");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "batchNormalization", "BatchNormalization");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "layerNormalization", "LayerNormalization");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "relu", "Relu");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "sigmoid", "Sigmoid");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "reduceSum", "ReduceSum");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "reduceMean", "ReduceMean");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "equal", "Equal");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "greater", "Greater");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "lesser", "Less");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "logicalNot", "Not");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "quantizeLinear", "QuantizeLinear");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "triangular", "Trilu");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "prelu", "PRelu");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "clamp", "Clip");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "gemm", "Gemm");

        Self { webnn_to_onnx, onnx_to_webnn }
    }

    pub fn webnn_to_onnx(&self, webnn_op: &str) -> Option<&str> {
        let key = webnn_op.to_ascii_lowercase();
        self.webnn_to_onnx.get(&key).map(|s| s.as_str())
    }

    pub fn onnx_to_webnn(&self, onnx_op: &str) -> Option<&str> {
        let key = onnx_op.to_ascii_lowercase();
        self.onnx_to_webnn.get(&key).map(|s| s.as_str())
    }
}

fn add_mapping(
    webnn_to_onnx: &mut HashMap<String, String>,
    onnx_to_webnn: &mut HashMap<String, String>,
    webnn: &str,
    onnx: &str,
) {
    webnn_to_onnx.insert(webnn.to_ascii_lowercase(), onnx.to_string());
    onnx_to_webnn.insert(onnx.to_ascii_lowercase(), webnn.to_string());
}

static MAPPER: Lazy<OpNameMapper> = Lazy::new(OpNameMapper::new);

pub fn mapper() -> &'static OpNameMapper {
    &MAPPER
}
