# TP1 - 8 images differentes a chaque lancement (graines aleatoires uniques)
# Consigne : 3 simples, 3 chargees, 2 difficiles (noms de fichiers = repere rapport)
# Si une image ne convient pas, remplace le fichier ou relance le script.
#
# Usage :
#   cd "...\TP1\data\images"
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\download_random_dataset.ps1

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot

# Graine unique par execution (camarades = images differentes)
$sessionId = "{0}{1}" -f [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds(), (Get-Random -Minimum 100000 -Maximum 999999)

function Save-Picsum {
    param(
        [string]$Seed,
        [string]$OutFile,
        [int]$W = 900,
        [int]$H = 675
    )
    $url = "https://picsum.photos/seed/$Seed/$W/$H"
    $path = Join-Path $here $OutFile
    curl.exe -sSL -L -o $path $url
    if (-not (Test-Path $path) -or ((Get-Item $path).Length -lt 2000)) {
        throw "Echec telechargement : $OutFile ($url)"
    }
}

Write-Host "Session (graine) : $sessionId" -ForegroundColor Cyan
Write-Host "Dossier : $here`n"

$map = @(
    @{ Name = "simple_01.jpg"; Seed = "tp1-$sessionId-s1" },
    @{ Name = "simple_02.jpg"; Seed = "tp1-$sessionId-s2" },
    @{ Name = "simple_03.jpg"; Seed = "tp1-$sessionId-s3" },
    @{ Name = "busy_01.jpg";   Seed = "tp1-$sessionId-b1" },
    @{ Name = "busy_02.jpg";   Seed = "tp1-$sessionId-b2" },
    @{ Name = "busy_03.jpg";   Seed = "tp1-$sessionId-b3" },
    @{ Name = "hard_01.jpg";   Seed = "tp1-$sessionId-h1" },
    @{ Name = "hard_02.jpg";   Seed = "tp1-$sessionId-h2" }
)

foreach ($m in $map) {
    Write-Host "  $($m.Name) ..."
    Save-Picsum -Seed $m.Seed -OutFile $m.Name
}

Write-Host ""
Write-Host "OK - images telechargees (picsum, aleatoire par seed) :" -ForegroundColor Green
Get-ChildItem $here -Filter "*.jpg" | Sort-Object Name | Format-Table Name, @{L='Ko';E={[math]::Round($_.Length/1kb,1)}}

Write-Host ""
Write-Host "Rappel : verifie que simple/busy/hard collent au TP ; sinon change 1-2 fichiers a la main." -ForegroundColor Yellow
Write-Host ""
