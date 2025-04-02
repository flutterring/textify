import torch
import whisper
import onnx

# Load Whisper model
model = whisper.load_model("large")

# Dummy input (for tracing)
dummy_input = torch.randn(1, 80, 3000)  # Mel spectrogram shape

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "whisper.onnx",
    opset_version=13,
    input_names=["mel"],
    output_names=["logits"]
)
