$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$destination = Join-Path $root '.tools\SurgeXT-1.3.4'
$archive = Join-Path $root '.tools\surge-xt-win64-1.3.4-portable-install.zip'
$url = 'https://github.com/surge-synthesizer/releases-xt/releases/download/1.3.4/surge-xt-win64-1.3.4-portable-install.zip'

if (Test-Path $destination) {
    throw "Destination already exists: $destination. Inspect it before reinstalling."
}
New-Item -ItemType Directory -Force (Split-Path $archive -Parent) | Out-Null
Invoke-WebRequest -Uri $url -OutFile $archive
$hash = (Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant()
Expand-Archive -LiteralPath $archive -DestinationPath $destination
@{
    version = '1.3.4'
    source_url = $url
    archive_sha256 = $hash
    installed_utc = [DateTime]::UtcNow.ToString('o')
    integrity_note = 'Local download hash; upstream release API does not provide a digest.'
} | ConvertTo-Json | Set-Content (Join-Path $destination 'install.json') -Encoding utf8
Remove-Item -LiteralPath $archive
Write-Output "Installed official portable Surge XT 1.3.4 to $destination (SHA256 $hash)"
