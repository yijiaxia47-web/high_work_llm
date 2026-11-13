@echo off
cd /d "%~dp0.."
python scripts\run_ablation.py --image-dir data\datasets\work_at_height\images\valid --outdir outputs\raw_baseline_001 --mode base --limit 20 %*
exit /b %errorlevel%
