# ============================================================
# SAH-RAG Ablation: Master GPU Execution Script
# Run on GPU server after cloning project + setting env vars.
# Usage: .\ablation\run_gpu_all.ps1
# ============================================================
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"

$sah = "C:\Users\USER1\Documents\SAH-RAG"
$py  = "$sah\.venv\Scripts\python.exe"
$startTime = Get-Date

Write-Host "=========================================================="
Write-Host " SAH-RAG Ablation: Master GPU Run"
Write-Host " Start: $startTime"
Write-Host "=========================================================="
Write-Host ""

Push-Location $sah
try {
    Write-Host ">>> STEP 1/5: Encode Tier-1 chunks (59 variants × datasets)"
    cmd /c "`"$py`" -m ablation.embedding.encode_chunks --tier 1 --batch-size 32 2>&1"
    if ($LASTEXITCODE -ne 0) { throw "encode_chunks tier1 failed" }

    Write-Host ""
    Write-Host ">>> STEP 2/5: Encode Tier-3 chunks (alt embedder, 10 pairs)"
    cmd /c "`"$py`" -m ablation.embedding.encode_chunks --tier 3 --batch-size 32 2>&1"
    if ($LASTEXITCODE -ne 0) { throw "encode_chunks tier3 failed" }

    Write-Host ""
    Write-Host ">>> STEP 3/5: Encode queries (all 17 datasets)"
    cmd /c "`"$py`" -m ablation.embedding.encode_queries --all 2>&1"
    if ($LASTEXITCODE -ne 0) { throw "encode_queries failed" }

    Write-Host ""
    Write-Host ">>> STEP 4/5: Run Tier-1 evaluations (49 cells)"
    cmd /c "`"$py`" -m ablation.run_all --tier 1 2>&1"
    if ($LASTEXITCODE -ne 0) { throw "run_all tier1 failed" }

    Write-Host ""
    Write-Host ">>> STEP 5/5: Run Tier-2 evaluations (272 free re-ranks)"
    cmd /c "`"$py`" -m ablation.run_all --tier 2 2>&1"
    if ($LASTEXITCODE -ne 0) { throw "run_all tier2 failed" }

    Write-Host ""
    Write-Host ">>> FINAL: Generate reports"
    cmd /c "`"$py`" -m ablation.reporting.all 2>&1"
} finally {
    Pop-Location
}

$elapsed = (Get-Date) - $startTime
Write-Host ""
Write-Host "=========================================================="
Write-Host " DONE in $($elapsed.ToString('hh\:mm\:ss'))"
Write-Host " Reports: D:\rag_work\ablation\reports\"
Write-Host "=========================================================="