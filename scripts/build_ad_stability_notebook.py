"""Build the self-contained Colab handoff for an unpublished local Git bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]
def cell(kind,text):
    result={"cell_type":kind,"metadata":{},"source":textwrap.dedent(text).strip()+"\n"}
    if kind=="code":result.update(execution_count=None,outputs=[])
    return result

def build():
    config_hash=hashlib.sha256((ROOT/"configs/pde_generalization/advection_diffusion_v2_stable.yaml").read_bytes().replace(b"\r\n",b"\n")).hexdigest()
    cells=[cell("markdown","""
    # Advection–diffusion V2 stability revision — five matched seeds
    New protocol `ad_v2_stable_v1`; historical outcomes are preserved. Seeds 0–4,
    both safeguarded Vanilla and VARA. 4,000 committed Adam steps, 500 warmup,
    seven 500-step blocks and 25-step matched probes. Fixed last state, no L-BFGS.
    This is an experiment, not a promise of a positive result. Shared optimizer
    improvements must not be attributed solely to the allocation controller.
    Select **Runtime → Change runtime type → GPU**. Upload the provided local
    `ad_v2_stability_revision.bundle` in the first cell. No GitHub push is needed.
    Run cells in order. Storage/checkpoints persist in Drive; rerunning the study
    cell resumes the latest complete block and preserves interrupted artifacts.
    Progress/ETA track the committed budget; discarded probes/replays add work.
    """),cell("code","""
    from google.colab import files
    from pathlib import Path
    uploaded=files.upload()
    bundle_names=[name for name in uploaded if name.endswith('.bundle')]
    assert len(bundle_names)==1, 'Upload ad_v2_stability_revision.bundle'
    BUNDLE=Path('/content')/bundle_names[0]
    assert BUNDLE.is_file()
    """),cell("code","""
    import subprocess, sys, torch
    subprocess.run([sys.executable,'-m','pip','install','-q','numpy','pandas','matplotlib','PyYAML','scipy','pytest','tqdm'],check=True)
    assert torch.cuda.is_available(), 'Enable a GPU runtime before running the primary study'
    print('GPU:',torch.cuda.get_device_name(0),'Torch:',torch.__version__,'CUDA:',torch.version.cuda)
    """),cell("code",f"""
    import hashlib, os
    SOURCE=Path('/content/VARA_PINN_AD_stability')
    HISTORICAL='cdb27c8be0e681654d8c6c3005d9658c0ae6ee73'
    BRANCH='codex/ad-v2-stability-revision'
    heads=subprocess.check_output(['git','bundle','list-heads',str(BUNDLE)],text=True).strip().splitlines()
    matches=[line.split()[0] for line in heads if line.split()[1]=='refs/heads/'+BRANCH]
    assert len(matches)==1, 'Unexpected local bundle branch'
    REVISION=matches[0]
    if not SOURCE.exists():
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/Arawal123/VARA-PINN-v3.git',str(SOURCE)],check=True)
        subprocess.run(['git','-C',str(SOURCE),'fetch','--no-tags','origin',HISTORICAL],check=True)
        subprocess.run(['git','-C',str(SOURCE),'checkout','--detach',HISTORICAL],check=True)
        subprocess.run(['git','-C',str(SOURCE),'bundle','verify',str(BUNDLE)],check=True)
        subprocess.run(['git','-C',str(SOURCE),'fetch',str(BUNDLE),'refs/heads/'+BRANCH+':refs/heads/'+BRANCH],check=True)
        subprocess.run(['git','-C',str(SOURCE),'checkout',BRANCH],check=True)
    os.chdir(SOURCE)
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==REVISION
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    cfg=SOURCE/'configs/pde_generalization/advection_diffusion_v2_stable.yaml'
    assert hashlib.sha256(cfg.read_bytes().replace(b'\\r\\n',b'\\n')).hexdigest()=='{config_hash}'
    print('Local experiment revision:',REVISION)
    """),cell("code","""
    from google.colab import drive
    drive.mount('/content/drive')
    RUN_LABEL='ad_v2_stable_v1_5seed_trial_01'  # Change only for a distinct new study.
    DRIVE_ROOT=Path('/content/drive/MyDrive/VARA_PINN_AD_stability')
    OUTPUT=DRIVE_ROOT/'runs'/RUN_LABEL
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    SEEDS=[0,1,2,3,4]
    print('Persistent results:',OUTPUT)
    """),cell("code","""
    subprocess.run([sys.executable,'-B','-m','pytest','tests/test_ad_stability_revision.py',
                    'tests/test_ad_causal_diagnostic.py','-q','-p','no:cacheprovider',
                    '--basetemp=/content/ad_stability_cpu_tests'],check=True)
    """),cell("code","""
    import json, time
    command=[sys.executable,'-u','-B','scripts/run_ad_v2_stability.py','--device','cuda',
             '--seeds',*[str(s) for s in SEEDS],'--output-dir',str(OUTPUT),'--resume']
    started=time.monotonic()
    log_path=OUTPUT.parent/(RUN_LABEL+'_console_'+time.strftime('%Y%m%d_%H%M%S')+'.log')
    with log_path.open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
            if line.startswith('@@PROGRESS '):
                p=json.loads(line[len('@@PROGRESS '):])
                run_index=2*SEEDS.index(p['seed'])+(p['mode']=='vara_v2')
                fraction=(run_index+p['step']/p['total_steps'])/(2*len(SEEDS))
                elapsed=time.monotonic()-started
                eta=elapsed*(1-fraction)/fraction if fraction>0 else float('nan')
                print(f"Committed-budget progress: {100*fraction:.2f}% | ETA estimate: {eta/60:.1f} min | physical Adam calls in run: {p['optimizer_calls']}",flush=True)
        returncode=process.wait()
    if returncode:raise subprocess.CalledProcessError(returncode,command)
    print('Completed and automatically verified all ten runs. Console:',log_path)
    """),cell("code","""
    import shutil
    PACKAGE_PARENT=Path('/content')/('ad_stability_package_'+time.strftime('%Y%m%d_%H%M%S'))
    PACKAGE_PARENT.mkdir()
    PACKAGE=PACKAGE_PARENT/'supplement_advection_diffusion_v2_stability_5seed'
    subprocess.run([sys.executable,'-B','scripts/package_ad_v2_stability.py',
                    '--results',str(OUTPUT),'--output',str(PACKAGE),
                    '--console-logs',*[str(p) for p in sorted(OUTPUT.parent.glob(RUN_LABEL+'_console_*.log'))]],check=True)
    ZIP=PACKAGE.with_suffix('.zip')
    subprocess.run([sys.executable,'-B','scripts/verify_ad_stability_zip.py',str(ZIP)],check=True)
    PERSISTENT_ZIP=DRIVE_ROOT/'packages'/PACKAGE_PARENT.name/ZIP.name
    PERSISTENT_ZIP.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ZIP,PERSISTENT_ZIP)
    print('ZIP:',ZIP,'Persistent ZIP:',PERSISTENT_ZIP,sep='\\n')
    """),cell("code","""
    files.download(str(ZIP))
    """)]
    return {"nbformat":4,"nbformat_minor":5,"metadata":{"colab":{"name":"Advection_Diffusion_V2_Stability_5Seed.ipynb"},"accelerator":"GPU",
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"cells":[{**c,"id":f"ad-stability-{i:02d}"} for i,c in enumerate(cells)]}

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",required=True)
    path=Path(parser.parse_args().output);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(str(path))
    path.write_text(json.dumps(build(),indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(path)
