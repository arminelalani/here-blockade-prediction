# Publishing this repository

1. Confirm that publicly sharing project code, student work, and the trained model is permitted by your course and teammates. A private GitHub repository is the safest initial choice.
2. Download `hERG_GitHub_Repository_Starter.zip`, unzip it on your Mac, and review the notebooks for any personal or restricted data. The files are arranged, **not refactored or fully re-executed**.
3. From JupyterLab, download the actual final **`Part1_Armineh-Rezagholilalani.onnx`** file and place it in the repository root if model sharing is permitted. The starter ZIP cannot contain that artifact because it was not uploaded here.
4. `.gitignore` deliberately excludes `*.onnx`; to version this specific approved artifact, use `git add -f Part1_Armineh-Rezagholilalani.onnx` or remove that one ignore rule. Verify size and integrity.
5. Do not include `hw1data.tab`, the assignment PDF, private paths, authentication tokens, downloaded ChemBERTa weights, `.venv`, offline wheel folders, or research caches.
6. Verify the Hugging Face Space link in README. If the Space is private or moved, update the link.
7. Create a GitHub repo named `herg-blockade-prediction` (prefer **Private** until permissions are clear). Upload the contents of this starter folder, not the ZIP file itself.
8. For the course submission, keep the separately validated `Part 2.zip`, the exact `.onnx`, and `TestModel.py` in your UCSF working directory; rerun the official checker after updating the course README.
9. Important: this starter copy sets `NAME = "Armineh-Rezagholilalani"` to match the successful run. Verify team naming rules before any official hand-in.

Optional local Git commands (after creating an empty remote repository):

```bash
git init
git add README.md requirements.txt .gitignore GITHUB_SETUP.md Part1_RF_HYBRID_FINAL.py app.py notebooks docs
git commit -m "Document hERG research and deployable RF-Hybrid model"
git branch -M main
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```
