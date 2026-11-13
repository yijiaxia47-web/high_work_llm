@echo off
cd /d "%~dp0.."
python scripts\prepare_evidence_from_yolo_labels.py --dataset-root data\datasets\work_at_height --split valid --outdir data\evidence\oracle_regenerated --limit 20 --min-classes 2 %*
exit /b %errorlevel%
