# AI Utopia — one-shot launcher: starts the FastAPI backend (correct AIUTOPIA_ROOT),
# waits for it to be healthy, then launches the desktop EXE.
$ErrorActionPreference = "SilentlyContinue"
$repo = "C:\Users\Carte\OneDrive\Desktop\AiUtopia"
$exe  = "$repo\gui\src-tauri\target\release\aiutopia-control-center.exe"

# 1. Free port 8777 if something's already on it
Get-NetTCPConnection -LocalPort 8777 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }

# 2. Start the backend with the repo as AIUTOPIA_ROOT + src on PYTHONPATH
$env:AIUTOPIA_ROOT = $repo
$env:PYTHONPATH    = "$repo\src"
$env:PYTHONIOENCODING = "utf-8"
$api = Start-Process -FilePath "py" -ArgumentList "-3.11","-m","aiutopia.api" `
        -WorkingDirectory $repo -PassThru -WindowStyle Hidden
Write-Host "[launch] backend pid=$($api.Id) — waiting for http://127.0.0.1:8777/api/health ..."

# 3. Poll health up to ~20s
$ok = $false
for ($i = 0; $i -lt 40; $i++) {
  try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:8777/api/health" -TimeoutSec 2 -UseBasicParsing
    if ($r.StatusCode -eq 200) { $ok = $true; break }
  } catch { Start-Sleep -Milliseconds 500 }
}
if ($ok) { Write-Host "[launch] backend healthy" } else { Write-Host "[launch] backend not responding (GUI will show offline / sample data)" }

# 4. Launch the desktop app
Start-Process -FilePath $exe
Write-Host "[launch] AI Utopia desktop app started"
