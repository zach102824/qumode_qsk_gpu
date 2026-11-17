"""Build the PRB-style methods PDF using a real LaTeX engine."""
import argparse,os,shutil,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--engine',help='Path to tectonic or pdflatex')
    args=p.parse_args();here=Path(__file__).resolve().parent
    engine=args.engine or shutil.which('tectonic') or shutil.which('pdflatex')
    if not engine and Path('/tmp/nov17_tex_bin/tectonic').exists():engine='/tmp/nov17_tex_bin/tectonic'
    if not engine:raise SystemExit('Install Tectonic or TeX Live with REVTeX 4.2, then rerun with --engine.')
    env=os.environ.copy();env.setdefault('XDG_CACHE_HOME','/tmp/nov17_tex_cache')
    if 'tectonic' in Path(engine).name:
        subprocess.run([engine,'--keep-logs','nov17_methods.tex'],cwd=here,env=env,check=True)
    else:
        latex=[engine,'-interaction=nonstopmode','-halt-on-error','nov17_methods.tex']
        subprocess.run(latex,cwd=here,env=env,check=True)
        subprocess.run(['bibtex','nov17_methods'],cwd=here,env=env,check=True)
        subprocess.run(latex,cwd=here,env=env,check=True)
        subprocess.run(latex,cwd=here,env=env,check=True)
    print(here/'nov17_methods.pdf')

if __name__=='__main__':main()
