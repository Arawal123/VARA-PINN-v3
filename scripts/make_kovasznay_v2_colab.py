"""Generate a self-contained Colab launcher pinned to an implemented Git SHA."""
import argparse
import ast
import json
from pathlib import Path
import re

def notebook(revision):
    cells=[]
    def md(text):cells.append({"cell_type":"markdown","metadata":{},"source":text.strip().splitlines(True)})
    def code(text):
        text=text.strip().replace("__SOURCE_REVISION__",revision)+"\n";ast.parse(text)
        cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":text.splitlines(True)})
    md("""# Kovasznay Re40 — reference-free VARA V2 / matched Vanilla
New experiment: five seeds0–4,4000 committed Adam steps, frozen1000 permitted interior observations, fixed boundary data, full V2 mechanisms. Historical V1 values are never substituted. Primary outcomes remain PENDING until this notebook actually finishes all10 GPU runs. Tiny CPU smoke outputs use seed999 and a separate directory. Follow cells in order; the full-study cell is explicit. Resume uses atomic completed-state checkpoints and preserves all failed/discarded/replayed work. The source is pinned, rather than floating main. See the frozen protocol and parity matrix.
""")
    code('''
import os, sys, json, subprocess, hashlib, pathlib, datetime, time, shutil
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
SOURCE_SHA = "__SOURCE_REVISION__"
REPOSITORY = "https://github.com/Arawal123/VARA-PINN-v3.git"
PROJECT = pathlib.Path("/content/VARA-PINN-Kovasznay-V2")
if not PROJECT.exists():
    subprocess.run(["git", "clone", "--no-checkout", REPOSITORY, str(PROJECT)], check=True)
    subprocess.run(["git", "fetch", "origin", SOURCE_SHA], cwd=PROJECT, check=True)
    subprocess.run(["git", "checkout", "--detach", SOURCE_SHA], cwd=PROJECT, check=True)
actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
assert actual == SOURCE_SHA, f"Existing checkout is {actual}; use a fresh project directory for another revision."
assert not subprocess.check_output(["git", "--no-optional-locks", "status", "--porcelain"], cwd=PROJECT, text=True).strip()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy", "pandas", "scipy", "matplotlib", "pyyaml", "tqdm", "pytest", "jinja2"], check=True)
os.chdir(PROJECT)
print("Frozen source SHA:", actual)
''')
    code('''
USE_GOOGLE_DRIVE = True  # Recommended: checkpoints and final ZIP survive runtime loss.
if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    STORAGE = pathlib.Path("/content/drive/MyDrive/VARA_PINN_Kovasznay_V2")
else:
    STORAGE = pathlib.Path("/content/VARA_PINN_Kovasznay_V2")
    print("Local /content storage is temporary. Download or copy the ZIP before ending the runtime.")
STUDY = STORAGE / ("re40_5seed_" + SOURCE_SHA[:12])
STUDY.mkdir(parents=True, exist_ok=True)
PROTOCOL = json.loads((PROJECT / "PROTOCOL_FROZEN.json").read_text())
marker = STUDY / "study_identity.json"
identity = {"source_sha": SOURCE_SHA, "protocol_hash": PROTOCOL["profiles"]["prescribed_sparse_primary"]["config_hash"]}
if marker.exists():
    assert json.loads(marker.read_text()) == identity, "Persistent study belongs to another source/protocol."
else:
    marker.write_text(json.dumps(identity, indent=2))
import torch
ATTEMPT = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
environment = {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
               "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "source_sha": SOURCE_SHA, "attempt": ATTEMPT}
env_folder = STUDY / "launcher_attempts" / ATTEMPT
env_folder.mkdir(parents=True, exist_ok=False)
(env_folder / "environment.json").write_text(json.dumps(environment, indent=2))
versions = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
(env_folder / "pip_freeze.txt").write_text(versions)
if not (STUDY / "environment.json").exists():
    (STUDY / "environment.json").write_text(json.dumps(environment, indent=2))
    (STUDY / "pip_freeze.txt").write_text(versions)
print(json.dumps(environment, indent=2)); print("Persistent study:", STUDY)
''')
    code('''
preflight = subprocess.run([sys.executable, "-B", "scripts/run_kovasznay_v2_publication.py", "--preflight",
                           "--config", "configs/kovasznay_v2_publication.yaml"], check=True, capture_output=True, text=True)
preflight_data = json.loads(preflight.stdout)
assert preflight_data["valid"] and preflight_data["source"]["git_commit"] == SOURCE_SHA
assert preflight_data["protocol_hash"] == identity["protocol_hash"]
if not (STUDY / "preflight.json").exists():
    (STUDY / "preflight.json").write_text(preflight.stdout)
print("Preflight passed. Full GPU outcomes:", preflight_data["primary_study_status"])
print((PROJECT / "V2_PARITY_MATRIX.md").read_text())
''')
    md("## Validate the implementation before primary training")
    code('''
test_root = STUDY / "tests"
test_root.mkdir(exist_ok=True)
test_attempt = test_root / ATTEMPT
test_attempt.mkdir(exist_ok=True)
junit = test_attempt / "junit.xml"
test_command = [sys.executable, "-B", "-m", "pytest", "tests/test_kovasznay_v2_publication.py",
                "-q", "-p", "no:cacheprovider", "--basetemp", "/content/kova_test_tmp_" + ATTEMPT,
                "--junitxml", str(junit)]
result = subprocess.run(test_command, capture_output=True, text=True)
(test_attempt / "stdout.txt").write_text(result.stdout)
(test_attempt / "stderr.txt").write_text(result.stderr)
print(result.stdout); print(result.stderr)
assert result.returncode == 0, "Scientific regression tests failed. Primary study must not start."
from src.training.kovasznay_v2_publication import source_info
evidence = {"exit_code": result.returncode, "command": test_command,
            "junit_sha256": hashlib.sha256(junit.read_bytes()).hexdigest(),
            "scientific_files": source_info()["scientific_files"]}
(test_attempt / "evidence.json").write_text(json.dumps(evidence, indent=2))
if not (test_root / "evidence.json").exists():
    shutil.copy2(junit, test_root / "junit.xml")
    shutil.copy2(test_attempt / "stdout.txt", test_root / "stdout.txt")
    shutil.copy2(test_attempt / "stderr.txt", test_root / "stderr.txt")
    shutil.copy2(test_attempt / "evidence.json", test_root / "evidence.json")
else:
    old = json.loads((test_root / "evidence.json").read_text())
    assert old["scientific_files"] == evidence["scientific_files"] and old["exit_code"] == 0
print("Required tests passed; their exact outputs are persistent.")
''')
    code('''
from tqdm.auto import tqdm
import pandas as pd
from IPython.display import display, clear_output

def run_live(command, log_path, total, description):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x") as log, tqdm(total=total, desc=description, unit="committed step", leave=True) as bar:
        process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        recent = []
        for line in process.stdout:
            log.write(line); log.flush(); recent.append(line); recent = recent[-25:]
            if line.startswith("@@PROGRESS "):
                progress = json.loads(line[len("@@PROGRESS "):])
                bar.update(max(0, int(progress["step"]) - bar.n))
                bar.set_postfix(phase=progress["phase"], calls=progress["optimizer_calls"],
                                ETA=f"{progress['eta_seconds'] / 60:.1f} min")
        code = process.wait()
        if code:
            print("".join(recent)); raise subprocess.CalledProcessError(code, command)

def command_for(seed, mode, folder, smoke=False):
    command = [sys.executable, "-u", "-B", "scripts/run_kovasznay_v2_publication.py",
               "--config", "configs/kovasznay_v2_publication.yaml", "--mode", mode, "--seed", str(seed),
               "--device", "cpu" if smoke else "cuda", "--output-dir", str(folder)]
    if smoke: command.append("--smoke")
    elif folder.exists() and any(folder.iterdir()): command.append("--resume")
    return command

def verify_one(folder):
    subprocess.run([sys.executable, "-B", "scripts/verify_kovasznay_v2_publication.py",
                    "--run-dir", str(folder), "--strict"], cwd=PROJECT, check=True)

def completed_table():
    rows = []
    for seed in range(5):
        for mode in ["vanilla", "vara_v2"]:
            folder = STUDY / f"seed_{seed}" / mode
            if (folder / "COMPLETE.json").exists():
                value = json.loads((folder / "summary.json").read_text())
                rows.append({"seed": seed, "mode": mode, "velocity_rel_l2": value["metrics"]["velocity_rel_l2"],
                             "u_rel_l2": value["metrics"]["u_rel_l2"], "v_rel_l2": value["metrics"]["v_rel_l2"],
                             "p_rel_l2_centered": value["metrics"]["p_rel_l2_centered"], "wall_seconds": value["wall_seconds"],
                             "committed_steps": value["committed_steps"], "optimizer_calls": value["optimizer_calls"]})
    return pd.DataFrame(rows)
''')
    code('''
SMOKE = STORAGE / ("smoke_" + ATTEMPT)
for mode in ["vanilla", "vara_v2"]:
    folder = SMOKE / "seed_999" / mode
    run_live(command_for(999, mode, folder, smoke=True), SMOKE / f"{mode}_stdout.log", 12, "CPU smoke " + mode)
    verify_one(folder)
print("Separate CPU smoke verified. No primary seed was trained by this cell.")
''')
    md("""## Start or resume the full five-seed GPU study
This cell executes the real10-run experiment. Select a GPU runtime first. It reuses verified complete results and resumes interrupted arms from complete-state checkpoints. A source/backend mismatch or uncertain interruption accounting stops strict verification; it never silently replaces or repairs data. Never merge seed999 smoke outputs into primary results.
""")
    code('''
assert torch.cuda.is_available(), "Select Runtime → Change runtime type → GPU; CPU fallback is prohibited."
from scripts.package_kovasznay_v2_publication import validate_tests
validate_tests(STUDY)
experiment_start = time.perf_counter()
invocations = []
with tqdm(total=10, desc="Five seeds × two methods", unit="run", leave=True) as overall:
    for seed in tqdm([0, 1, 2, 3, 4], desc="Seed", leave=False):
        for mode in ["vanilla", "vara_v2"]:
            folder = STUDY / f"seed_{seed}" / mode
            if (folder / "COMPLETE.json").exists():
                verify_one(folder)
                print(f"Verified completed seed{seed}/{mode}; skipped without overwrite.")
            else:
                command = command_for(seed, mode, folder)
                invocations.append(command)
                log_path = STUDY / "process_logs" / f"seed{seed}_{mode}_{ATTEMPT}.log"
                run_live(command, log_path, 4000, f"Seed{seed} {mode}")
                verify_one(folder)
            overall.update(1)
            table = completed_table()
            display(table)
            if len(table):
                paired = table.pivot(index="seed", columns="mode", values="velocity_rel_l2").dropna()
                if {"vanilla", "vara_v2"}.issubset(paired.columns):
                    paired["paired_improvement_percent"] = 100 * (paired.vanilla - paired.vara_v2) / paired.vanilla
                    display(paired)  # Includes every worse seed and negative improvement.
            print("This launcher elapsed minutes:", round((time.perf_counter() - experiment_start) / 60, 2))
(env_folder / "launcher_execution_log.json").write_text(json.dumps({"commands": invocations, "source_sha": SOURCE_SHA}, indent=2))
assert len(completed_table()) == 10
print("All ten primary runs complete and independently verified.")
''')
    md("## Build and independently verify the supplementary ZIP")
    code('''
ANALYSIS = STORAGE / ("analysis_" + SOURCE_SHA[:12] + "_" + ATTEMPT) / "supplement"
result = subprocess.run([sys.executable, "-B", "scripts/package_kovasznay_v2_publication.py",
                         "--input-root", str(STUDY), "--output-dir", str(ANALYSIS), "--strict"],
                        cwd=PROJECT, check=True, capture_output=True, text=True)
print(result.stdout)
package = json.loads(result.stdout)
ZIP = pathlib.Path(package["zip"])
subprocess.run([sys.executable, "-B", str(ANALYSIS / "verify_package.py"), "--zip", str(ZIP), "--strict"], check=True)
print((ANALYSIS / "provenance/verification_report.json").read_text())
display(pd.read_csv(ANALYSIS / "tables/per_seed_metrics.csv"))
display(pd.read_csv(ANALYSIS / "statistics/paired_statistics.csv"))
print("Final persistent ZIP:", ZIP)
print("SHA256:", package["sha256"])
from google.colab import files
files.download(str(ZIP))
''')
    for i,c in enumerate(cells):c["id"]=f"kova-v2-{i:02d}"
    return {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                                      "language_info":{"name":"python"},"colab":{"name":"kovasznay_v2_5seed_publication_colab.ipynb","provenance":[]}},
            "nbformat":4,"nbformat_minor":5}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--revision",required=True);p.add_argument("--output",default="notebooks/kovasznay_v2_5seed_publication_colab.ipynb")
    a=p.parse_args();assert re.fullmatch(r"[0-9a-f]{40}",a.revision),"Use an exact Git SHA"
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(notebook(a.revision),indent=1)+"\n",encoding="utf-8")
    print(output.resolve())
if __name__=="__main__":main()
