# ==========================================================================
#  Publica una nueva version de UPME AutoCarga.
#  - El ZIP de instalacion se reparte por Google Drive.
#  - GitHub solo sirve las actualizaciones de codigo (version.json + archivos).
#
#  Solo codigo (los usuarios se actualizan solos al abrir el programa):
#     .\publicar.ps1 -Version 1.0.1 -Notas "Corrige X"
#
#  Paquete completo (cambio en _runtime: librerias nuevas, Python, etc.).
#  Genera dist\UPME_AutoCarga_v<version>.zip y, antes de publicar en GitHub,
#  espera a que usted lo suba a la carpeta de Drive:
#     .\publicar.ps1 -Version 1.1.0 -Notas "Nueva libreria" -PaqueteCompleto
#
#  -UrlPaquete : enlace de la carpeta de Drive donde esta el ZIP. Se recuerda
#                entre versiones (queda en version.json); el programa lo abre
#                cuando un usuario necesita el paquete completo.
#  -SinCommit : solo prepara los archivos en el repo (para revisarlos).
#  -SinPush   : hace commit pero no lo sube.
#
#  Nota: este archivo es ASCII a proposito (PowerShell 5.1 lee los .ps1 sin
#  BOM como ANSI), por eso "El_Papa" se arma con [char]0xE1.
# ==========================================================================
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$Notas = "",
    [switch]$PaqueteCompleto,
    [switch]$SinCommit,
    [switch]$SinPush,
    [string]$UrlPaquete,
    [string]$Repo = (Join-Path $PSScriptRoot "..\FunctDesEPC")
)
$ErrorActionPreference = "Stop"

$CarpetaRepo = "UPME_AutoCarga"
$Dev = $PSScriptRoot
$Destino = Join-Path $Repo $CarpetaRepo
$Utf8 = New-Object System.Text.UTF8Encoding($false)

# Archivos que el actualizador descarga a cada PC.
$ArchivosApp = @(
    "iniciar.py", "actualizador.py", "config_usuario.py",
    "upme_autocarga_gui.py", "upme_autocarga_corregido.py",
    "procesar_formato3_p.py", "bizagi_selectors.py",
    "El_Pap$([char]0xE1).xlsx"
)
# Solo van al repo (documentacion / herramientas), no al actualizador.
$ArchivosRepo = @("README.md", "LEEME.txt", ".env.ejemplo", "publicar.ps1")
# Raiz del paquete (ejecutable firmado + DLLs de Python embebido). Los runtime de
# Visual C++ (vcruntime140*, msvcp140) van aqui para PCs sin VC++ Redistributable.
$ArchivosRaizZip = @("UPME AutoCarga.exe", "python3.dll", "python313.dll", "python313._pth",
                     "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "LEEME.txt", ".env.ejemplo")

# --- 0. Validaciones -------------------------------------------------------
if ($Version -notmatch '^\d+\.\d+\.\d+$') { throw "Version invalida '$Version' (use X.Y.Z)." }
if (-not (Test-Path (Join-Path $Repo ".git"))) { throw "No encuentro el repo en $Repo" }
foreach ($f in $ArchivosApp + $ArchivosRepo) {
    if (-not (Test-Path -LiteralPath (Join-Path $Dev $f))) { throw "Falta $f en $Dev" }
}
$VersionJsonDev = Join-Path $Dev "version.json"
$anterior = $null
if (Test-Path $VersionJsonDev) { $anterior = Get-Content $VersionJsonDev -Raw -Encoding UTF8 | ConvertFrom-Json }
if ($anterior -and [version]$Version -le [version]$anterior.version) {
    throw "La version $Version no es mayor que la actual ($($anterior.version))."
}

git -C $Repo pull --ff-only
if ($LASTEXITCODE) { throw "git pull fallo en $Repo" }

# --- 1. Copiar archivos al repo -------------------------------------------
New-Item -ItemType Directory -Force $Destino | Out-Null
foreach ($f in $ArchivosApp + $ArchivosRepo) {
    Copy-Item -LiteralPath (Join-Path $Dev $f) -Destination (Join-Path $Destino $f) -Force
}
Copy-Item (Join-Path $Dev "_runtime\sitecustomize.py") (Join-Path $Destino "runtime_sitecustomize.py") -Force
# Sin conversion de fin de linea: los SHA-256 deben coincidir byte a byte.
[IO.File]::WriteAllText((Join-Path $Destino ".gitattributes"), "* -text`n", $Utf8)
[IO.File]::WriteAllText((Join-Path $Destino ".gitignore"), ".env`n__pycache__/`n", $Utf8)

# --- 2. version.json ------------------------------------------------------
$hashes = [ordered]@{}
foreach ($f in $ArchivosApp) {
    $hashes[$f] = (Get-FileHash -LiteralPath (Join-Path $Dev $f) -Algorithm SHA256).Hash.ToLower()
}
$esPaquete = $PaqueteCompleto -or -not $anterior
$paqueteMinimo = if ($esPaquete) { $Version } else { $anterior.paquete_minimo }
if (-not $UrlPaquete -and $anterior) { $UrlPaquete = $anterior.url_paquete }
if (-not $UrlPaquete) { $UrlPaquete = "" }
$info = [ordered]@{
    version        = $Version
    notas          = $Notas
    fecha          = (Get-Date -Format "yyyy-MM-dd")
    paquete_minimo = $paqueteMinimo
    url_paquete    = $UrlPaquete
    archivos       = $hashes
}
$json = $info | ConvertTo-Json -Depth 5
[IO.File]::WriteAllText($VersionJsonDev, $json, $Utf8)
[IO.File]::WriteAllText((Join-Path $Destino "version.json"), $json, $Utf8)
Write-Host "version.json -> $Version (paquete minimo $paqueteMinimo)"

# --- 3. ZIP del paquete completo ------------------------------------------
$zip = $null
if ($esPaquete) {
    $dist = Join-Path $Dev "dist"
    $stage = Join-Path $dist "UPME AutoCarga"
    if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
    New-Item -ItemType Directory -Force $stage | Out-Null

    foreach ($f in $ArchivosRaizZip + $ArchivosApp + @("version.json")) {
        Copy-Item -LiteralPath (Join-Path $Dev $f) -Destination (Join-Path $stage $f)
    }
    # _runtime sin cache, pip ni pruebas (~140 MB menos).
    robocopy (Join-Path $Dev "_runtime") (Join-Path $stage "_runtime") /E /NFL /NDL /NJH /NJS /NP `
        /XD __pycache__ pip "pip-*.dist-info" tests turtledemo /XF *.pyc desktop.ini | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy fallo ($LASTEXITCODE)" }
    $global:LASTEXITCODE = 0
    [IO.File]::WriteAllText((Join-Path $stage "_runtime\paquete.txt"), $Version, $Utf8)

    $zip = Join-Path $dist "UPME_AutoCarga_v$Version.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    # Se arma entrada por entrada: CreateFromDirectory en PowerShell 5.1 usa "\"
    # como separador, y el estandar ZIP (y muchos descompresores) exigen "/".
    Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
    $archivo = [IO.Compression.ZipFile]::Open($zip, [IO.Compression.ZipArchiveMode]::Create)
    try {
        $raiz = (Get-Item $dist).FullName.TrimEnd("\") + "\"
        foreach ($f in Get-ChildItem $stage -Recurse -File -Force) {
            $nombre = $f.FullName.Substring($raiz.Length).Replace("\", "/")
            [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archivo, $f.FullName, $nombre,
                [IO.Compression.CompressionLevel]::Optimal) | Out-Null
        }
    } finally { $archivo.Dispose() }
    Remove-Item $stage -Recurse -Force
    Write-Host ("ZIP: {0} ({1:N0} MB)" -f $zip, ((Get-Item $zip).Length / 1MB))
}

# --- 4. Commit y push -----------------------------------------------------
if ($SinCommit) { Write-Host "Archivos listos en $Destino (sin commit)."; return }
if ($zip -and $anterior) {
    # Al publicar, los usuarios con paquete viejo seran enviados a Drive:
    # el ZIP tiene que estar alli antes.
    Write-Host ""
    Write-Host "Suba $zip a la carpeta de Drive"
    if ($UrlPaquete) { Write-Host "  $UrlPaquete" }
    Read-Host "y presione Enter para publicar en GitHub (Ctrl+C para cancelar)" | Out-Null
}
git -C $Repo add -- $CarpetaRepo
git -C $Repo commit -m "UPME AutoCarga v$Version" -m $Notas
if ($LASTEXITCODE) { throw "git commit fallo" }
if ($SinPush) { Write-Host "Commit hecho (sin push)."; return }
git -C $Repo push
if ($LASTEXITCODE) { throw "git push fallo" }

if ($zip -and -not $anterior) { Write-Host "Suba $zip a la carpeta de Drive para repartirlo." }
Write-Host "Listo: v$Version publicada."
