"""
Script maestro: ejecuta todas las fases en orden.
Uso: python -X utf8 scripts/run_all.py [--from-phase N]
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCRIPTS = [
    ("00", ROOT / "scripts" / "00_data_cleaning.py"),
    ("01", ROOT / "scripts" / "01_eda.py"),
    ("02", ROOT / "scripts" / "02_modelos_predictivos.py"),
    ("03", ROOT / "scripts" / "03_clustering_recomendacion.py"),
]

start_from = "00"
if "--from-phase" in sys.argv:
    idx = sys.argv.index("--from-phase")
    start_from = sys.argv[idx + 1]

for phase_id, script_path in SCRIPTS:
    if phase_id < start_from:
        print(f"[SKIP] Fase {phase_id}")
        continue
    print(f"\n{'='*60}")
    print(f"  FASE {phase_id} — {script_path.name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(script_path)],
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else "ERROR"
    print(f"\n[{status}] Fase {phase_id} en {elapsed:.1f}s")
    if result.returncode != 0:
        print("Abortando.")
        sys.exit(1)

print("\n✓ Todas las fases completadas.")
print("Para lanzar la app: streamlit run app/app.py")
