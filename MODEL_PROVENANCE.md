# Model Provenance

The workflow uses `mobilenetv2-10.onnx`, the MobileNetV2 1.0 model from the
ONNX Model Zoo. The model is trained for the 1,000 ImageNet classes and is
distributed by the ONNX Model Zoo under Apache-2.0.

- Model repository: <https://huggingface.co/onnxmodelzoo/mobilenetv2-10>
- Pinned repository revision: `02be6d5da12f60afa5b76260fc44f9f715b4cf75`
- Model size: `13963115` bytes
- Model SHA-256: `0e7c0aa4bc74650386fa1d2c84705753de7c2bdb21909ada5c59154bb429e092`
- Upstream model documentation: <https://github.com/onnx/models/tree/4f43949841cb55a0b98dc8fcd045431ccafd9f96/validated/vision/classification/mobilenet>
- ImageNet labels revision: `4f43949841cb55a0b98dc8fcd045431ccafd9f96`
- Labels SHA-256: `acf75ef0abe89694b19056e0796401068b459c457baa30335f240c7692857355`

Floability downloads and verifies the model and labels on the submit host.
Workers receive those staged files from TaskVine and do not access the model
repository directly.
