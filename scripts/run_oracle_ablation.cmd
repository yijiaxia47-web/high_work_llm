@echo off
cd /d "%~dp0.."
python scripts\run_ablation.py --manifest data\evidence\oracle\manifest_with_gold.json --outdir outputs\oracle_ablation_001 --mode leave_one_out --repeats 3 %*
exit /b %errorlevel%
