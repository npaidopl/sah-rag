param(
    [string]$Model = "Qwen/Qwen2.5-32B-Instruct",
    [string]$Target = "D:\rag_work\models\Qwen2.5-32B-Instruct",
    [int]$MaxOuterRetries = 200,
    [int]$OuterRetryWaitSec = 30
)

$ErrorActionPreference = "Continue"
$logPath = Join-Path $Target "_download_progress.log"
$sessionStart = Get-Date

function Log-Message {
    param([string]$Msg, [string]$Color = "White")
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] $Msg" -ForegroundColor $Color
    $logDir = Split-Path $logPath -Parent
    if (Test-Path $logDir) {
        "[$ts] $Msg" | Out-File -FilePath $logPath -Append -Encoding utf8
    }
}

if (-not (Test-Path $Target)) {
    New-Item -ItemType Directory -Force -Path $Target | Out-Null
    Write-Host "Created target: $Target" -ForegroundColor Green
}

Log-Message "=== SESSION START ===" "Cyan"
Log-Message "Model: $Model" "Cyan"
Log-Message "Target: $Target" "Cyan"
Log-Message "Host: $env:COMPUTERNAME / User: $env:USERNAME" "Cyan"

$hasAria2 = [bool](Get-Command aria2c -ErrorAction SilentlyContinue)
Log-Message "aria2c available: $hasAria2" $(if ($hasAria2) { "Green" } else { "Yellow" })

Log-Message "Fetching file tree from HF API..." "Cyan"
try {
    $tree = Invoke-RestMethod -Uri "https://huggingface.co/api/models/$Model/tree/main?recursive=true" -TimeoutSec 60
} catch {
    Log-Message "ERROR: Cannot fetch tree: $_" "Red"
    exit 1
}

$files = $tree | Where-Object { $_.type -eq "file" -and $_.path -ne ".gitattributes" } | ForEach-Object {
    [PSCustomObject]@{ rfilename = $_.path; size = $_.size }
}
$totalBytes = ($files | Measure-Object -Property size -Sum).Sum
$totalGB = [math]::Round($totalBytes / 1GB, 2)
Log-Message "Files: $($files.Count), Total: $totalGB GB" "Cyan"

function Get-Progress {
    $localBytes = 0
    $completeCount = 0
    foreach ($f in $files) {
        $localPath = Join-Path $Target $f.rfilename
        if (Test-Path $localPath) {
            $actual = (Get-Item $localPath).Length
            if ($actual -eq $f.size) {
                $completeCount++
                $localBytes += $actual
            } else {
                $localBytes += $actual
            }
        }
    }
    $pct = if ($totalBytes -gt 0) { [math]::Round($localBytes / $totalBytes * 100, 2) } else { 0 }
    return [PSCustomObject]@{
        Complete = $completeCount
        Total = $files.Count
        LocalGB = [math]::Round($localBytes / 1GB, 2)
        TotalGB = $totalGB
        Percent = $pct
    }
}

$p = Get-Progress
Log-Message "Initial state: $($p.Complete)/$($p.Total) files, $($p.LocalGB)/$($p.TotalGB) GB ($($p.Percent)%)" "Yellow"

if ($p.Complete -eq $p.Total) {
    Log-Message "ALL FILES ALREADY COMPLETE - nothing to do" "Green"
    exit 0
}
$baseUrl = "https://huggingface.co/$Model/resolve/main"

foreach ($f in $files) {
    $localPath = Join-Path $Target $f.rfilename

    if (Test-Path $localPath) {
        $actual = (Get-Item $localPath).Length
        if ($actual -eq $f.size) {
            Log-Message "SKIP: $($f.rfilename) ($([math]::Round($actual/1MB,0)) MB)" "Green"
            continue
        }
        $filePct = [math]::Round($actual / $f.size * 100, 1)
        Log-Message "RESUME: $($f.rfilename) ($filePct%)" "Yellow"
    } else {
        Log-Message "NEW: $($f.rfilename) ($([math]::Round($f.size/1MB,0)) MB)" "Cyan"
    }

    $success = $false
    for ($attempt = 1; $attempt -le $MaxOuterRetries; $attempt++) {
        $url = "$baseUrl/$($f.rfilename)?download=true"

        if ($hasAria2) {
            aria2c -c -x 16 -s 16 -k 1M -d $Target -o $f.rfilename --auto-file-renaming=false --allow-overwrite=false --retry-wait=10 --max-tries=1 --timeout=120 --summary-interval=60 --console-log-level=warn $url 2>&1 | Out-Null
        } else {
            hf download $Model $f.rfilename --local-dir $Target 2>&1 | Out-Null
        }

        if (Test-Path $localPath) {
            $actual = (Get-Item $localPath).Length
            if ($actual -eq $f.size) {
                Log-Message "  DONE: $($f.rfilename) ($([math]::Round($actual/1MB,0)) MB)" "Green"
                $success = $true
                break
            } else {
                $filePct = [math]::Round($actual / $f.size * 100, 1)
                Log-Message "  Attempt $attempt : $filePct% ($([math]::Round($actual/1MB,0))/$([math]::Round($f.size/1MB,0)) MB)" "Yellow"
            }
        } else {
            Log-Message "  Attempt $attempt : not started" "Yellow"
        }

        Start-Sleep -Seconds $OuterRetryWaitSec
    }

    if (-not $success) {
        Log-Message "  FAILED after $MaxOuterRetries attempts: $($f.rfilename)" "Red"
        exit 2
    }
}

$p = Get-Progress
$elapsed = (Get-Date) - $sessionStart
Log-Message "=== SESSION COMPLETE ===" "Green"
Log-Message "Duration: $([math]::Round($elapsed.TotalMinutes,1)) min" "Green"
Log-Message "Final: $($p.Complete)/$($p.Total) files, $($p.LocalGB)/$($p.TotalGB) GB ($($p.Percent)%)" "Green"
