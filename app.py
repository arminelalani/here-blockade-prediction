"""CDD203 hERG blockade demo using the exact frozen Part 1 ONNX model."""
from pathlib import Path
import numpy as np
import onnxruntime as ort
import gradio as gr
from Part1_RF_HYBRID_FINAL import NAME, THRESHOLD, featurize

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / f"Part1_{NAME}.onnx"
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Missing {MODEL_PATH.name}; copy the ONNX from Part 1 here.")
SESSION = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
INPUT = SESSION.get_inputs()[0]
INPUT_DTYPE = {"tensor(double)": np.float64, "tensor(int64)": np.int64}.get(INPUT.type, np.float32)


def predict(smiles: str):
    smiles = (smiles or "").strip()
    if not smiles:
        return "Please enter a SMILES string.", None
    try:
        features = np.asarray(featurize([smiles]), dtype=INPUT_DTYPE)
        output = np.asarray(SESSION.run(None, {INPUT.name: features})[-1], dtype=float)
        probability = float(output[0, 1] if output.ndim == 2 and output.shape[1] == 2 else output.ravel()[0])
        if not np.isfinite(probability):
            raise ValueError("Model produced a non-finite prediction")
    except (ValueError, TypeError) as error:
        return f"Invalid or unsupported SMILES: {error}", None
    label = "Predicted hERG blocker" if probability >= THRESHOLD else "Predicted non-blocker"
    return f"{label} (threshold = {THRESHOLD:.2f})", round(probability, 6)


with gr.Blocks(title="hERG Blockade Predictor") as demo:
    gr.Markdown("# hERG Blockade Predictor\nRF-Hybrid (Morgan fingerprint + molecular descriptors), ONNX inference. Research/educational use only.")
    smiles = gr.Textbox(label="Molecule SMILES", placeholder="e.g. CCO", lines=2)
    button = gr.Button("Predict")
    result = gr.Textbox(label="Prediction")
    probability = gr.Number(label="P(hERG blocker)", precision=6)
    button.click(predict, inputs=smiles, outputs=[result, probability])
    smiles.submit(predict, inputs=smiles, outputs=[result, probability])
    gr.Examples(["CCO", "CC(=O)Oc1ccccc1C(=O)O", "CN(C)CCOC(c1ccccc1)c1ccccc1.Cl"], inputs=smiles)

if __name__ == "__main__":
    demo.launch()
