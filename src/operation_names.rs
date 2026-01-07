use std::collections::HashMap;

use once_cell::sync::Lazy;

#[derive(Debug, Clone)]
pub struct OpNameMapper {
    webnn_to_onnx: HashMap<String, String>,
    onnx_to_webnn: HashMap<String, String>,
}

impl Default for OpNameMapper {
    fn default() -> Self {
        Self::new()
    }
}

impl OpNameMapper {
    pub fn new() -> Self {
        let mut webnn_to_onnx = HashMap::new();
        let mut onnx_to_webnn = HashMap::new();

        // Matrix operations
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "matmul", "MatMul");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "gemm", "Gemm");

        // Convolution operations
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "conv2d", "Conv");
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "convTranspose2d",
            "ConvTranspose",
        );

        // Pooling operations
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "averagePool2d",
            "AveragePool",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "maxPool2d",
            "MaxPool",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "globalAveragePool",
            "GlobalAveragePool",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "globalMaxPool",
            "GlobalMaxPool",
        );

        // Normalization operations
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "batchNormalization",
            "BatchNormalization",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "layerNormalization",
            "LayerNormalization",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "instanceNormalization",
            "InstanceNormalization",
        );

        // Activation operations
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "relu", "Relu");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "sigmoid", "Sigmoid");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "tanh", "Tanh");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "softmax", "Softmax");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "prelu", "PRelu");
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "leakyRelu",
            "LeakyRelu",
        );
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "elu", "Elu");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "clamp", "Clip");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "gelu", "Gelu");
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "hardSigmoid",
            "HardSigmoid",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "hardSwish",
            "HardSwish",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "softplus",
            "Softplus",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "softsign",
            "Softsign",
        );

        // Elementwise binary operations
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "add", "Add");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "sub", "Sub");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "mul", "Mul");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "div", "Div");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "pow", "Pow");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "max", "Max");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "min", "Min");

        // Elementwise unary operations
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "abs", "Abs");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "ceil", "Ceil");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "cos", "Cos");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "exp", "Exp");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "floor", "Floor");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "log", "Log");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "neg", "Neg");
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "reciprocal",
            "Reciprocal",
        );
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "sin", "Sin");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "sqrt", "Sqrt");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "tan", "Tan");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "erf", "Erf");
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "identity",
            "Identity",
        );
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "sign", "Sign");

        // Reduction operations
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "reduceSum",
            "ReduceSum",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "reduceMean",
            "ReduceMean",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "reduceMax",
            "ReduceMax",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "reduceMin",
            "ReduceMin",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "reduceProduct",
            "ReduceProd",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "reduceL1",
            "ReduceL1",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "reduceL2",
            "ReduceL2",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "reduceLogSum",
            "ReduceLogSum",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "reduceLogSumExp",
            "ReduceLogSumExp",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "reduceSumSquare",
            "ReduceSumSquare",
        );

        // Comparison operations
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "equal", "Equal");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "greater", "Greater");
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "greaterOrEqual",
            "GreaterOrEqual",
        );
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "lesser", "Less");
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "lesserOrEqual",
            "LessOrEqual",
        );

        // Logical operations
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "logicalAnd", "And");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "logicalOr", "Or");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "logicalNot", "Not");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "logicalXor", "Xor");

        // Tensor manipulation operations
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "concat", "Concat");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "expand", "Expand");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "gather", "Gather");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "pad", "Pad");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "reshape", "Reshape");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "slice", "Slice");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "split", "Split");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "squeeze", "Squeeze");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "tile", "Tile");
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "transpose",
            "Transpose",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "unsqueeze",
            "Unsqueeze",
        );
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "where", "Where");
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "triangular",
            "Trilu",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "resample2d",
            "Resize",
        );

        // Quantization operations
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "quantizeLinear",
            "QuantizeLinear",
        );
        add_mapping(
            &mut webnn_to_onnx,
            &mut onnx_to_webnn,
            "dequantizeLinear",
            "DequantizeLinear",
        );

        // Recurrent operations
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "gru", "GRU");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "gruCell", "GRU");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "lstm", "LSTM");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "lstmCell", "LSTM");

        // Other operations
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "argMax", "ArgMax");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "argMin", "ArgMin");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "cast", "Cast");
        add_mapping(&mut webnn_to_onnx, &mut onnx_to_webnn, "flatten", "Flatten");

        Self {
            webnn_to_onnx,
            onnx_to_webnn,
        }
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
