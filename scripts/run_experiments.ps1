param(
    [Parameter(Mandatory)]
    [ValidateSet('smoke', 'probe1', 'probe2', 'abcd', 'fedmd', 'local', 'rahfl', 'aughfl')]
    [string] $Experiment,
    [string] $DataRoot,
    [string] $Python = 'python',
    [ValidateSet('auto', 'cpu', 'cuda')]
    [string] $Device = 'auto'
)
$ErrorActionPreference = 'Stop'
if ($DataRoot) { $DataRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($DataRoot) }
$pythonExe = (Get-Command $Python -CommandType Application).Source
$originalPath = $env:Path
$prefix = Split-Path -Parent $pythonExe
$library = Join-Path $prefix 'Library\bin'
if (Test-Path -LiteralPath $library) {
    $env:Path = "$library;$(Join-Path $prefix 'Scripts');$originalPath"
}
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $modules = @{
        smoke = 'scripts.smoke'; probe1 = 'probes.probe1_local_value'; probe2 = 'probes.probe2_freq_delta_r'
        abcd = 'methods.joint_framework'; fedmd = 'baselines.fedmd_kd'; local = 'baselines.local_ce'
        rahfl = 'baselines.rahfl_adapter'; aughfl = 'baselines.aughfl_adapter'
    }
    $options = @()
    if ($DataRoot) { $options += @('--data-root', $DataRoot) }
    $options += @('--device', $Device)
    if ($Experiment -eq 'probe2') { $options += '--prepare-checkpoints' }
    if ($Experiment -eq 'smoke') { $options += '--with-adapters' }
    & $pythonExe -m $modules[$Experiment] @options
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
    $env:Path = $originalPath
}
