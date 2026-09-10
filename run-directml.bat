@echo off
REM Maxed-out DirectML defaults for RX580 8GB + 12-core Intel CPU + 32GB RAM.
REM Thread/memory tuning also lives in code defaults (modules/core.py), so a
REM bare "run.py --execution-provider dml" is already maxed — these env vars
REM just make it explicit and cover libraries imported before our tuning runs.
set OMP_NUM_THREADS=10
set MKL_NUM_THREADS=10
set OPENBLAS_NUM_THREADS=10
set OMP_WAIT_POLICY=ACTIVE
set OMP_DYNAMIC=FALSE
set KMP_BLOCKTIME=0
python run.py --execution-provider dml
