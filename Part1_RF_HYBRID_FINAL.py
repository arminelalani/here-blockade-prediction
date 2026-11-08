"""CDD203 Homework 1 — frozen RF-Hybrid hERG blocker submission.

Before running: replace NAME with the group name required by your instructor.
Run: python Part1_RF_HYBRID_FINAL.py
Then: python TestModel.py Part1_RF_HYBRID_FINAL.py
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import to_onnx

# TODO: replace the placeholder with your ACTUAL group name before submission.
NAME = "Armineh-Rezagholilalani"
THRESHOLD = 0.49
SEED = 42

RDLogger.DisableLog('rdApp.warning')
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def featurize(smiles):
    """Fixed (n_molecules, 2058) float32 features; matches notebook 04 exactly."""
    output = np.zeros((len(smiles), 2058), dtype=np.float32)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {i}: {s!r}")
        DataStructs.ConvertToNumpyArray(GEN.GetFingerprint(mol), output[i, :2048])
        output[i, 2048:] = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            Chem.GetFormalCharge(mol),
            sum(atom.GetAtomicNum() == 7 for atom in mol.GetAtoms()),
        ]
    return output


def build_model():
    """Frozen model; no tuning on locked research or hidden test sets."""
    return RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=1,
        max_features=0.2,
        max_depth=30,
        class_weight='balanced',
        random_state=SEED,
        n_jobs=2,
    )


def save_onnx(model, sample):
    """No ZipMap: grader takes the last ONNX output as a numeric probability array."""
    path = Path(__file__).resolve().parent / f"Part1_{NAME}.onnx"
    converted = to_onnx(model, sample, options={'zipmap': False}, target_opset=12)
    path.write_bytes(converted.SerializeToString())
    print(f"ONNX saved: {path.name} ({path.stat().st_size / 1e6:.2f} MB)")
    return path


def find_data():
    here = Path(__file__).resolve().parent
    options = (here / 'data' / 'hw1data.tab', here / 'hw1data.tab')
    for path in options:
        if path.exists():
            return path
    raise FileNotFoundError("Put hw1data.tab beside this script or in a data/ subfolder")


def main():
    if NAME == 'GROUP-MEMBER-NAMES-HERE':
        raise ValueError('Fill in NAME with the exact group name before running.')
    data_path = find_data()
    df = pd.read_csv(data_path, sep='\t')
    if {'Drug', 'Y'}.issubset(df.columns):
        pass
    elif df.shape[1] == 3:
        df.columns = ['Drug_ID', 'Drug', 'Y']
    else:
        raise ValueError('Expected Drug and Y columns in hw1data.tab')
    X = featurize(df['Drug'].tolist())
    y = df['Y'].to_numpy(dtype=int)
    assert np.isfinite(X).all() and set(np.unique(y)) <= {0, 1}
    start = time.perf_counter()
    model = build_model().fit(X, y)
    print(f"Trained on {len(y)} molecules in {time.perf_counter()-start:.2f} s")
    save_onnx(model, X[:1])


if __name__ == '__main__':
    main()
