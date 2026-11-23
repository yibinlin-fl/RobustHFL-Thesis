param([string] $Proxy)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    New-Item -ItemType Directory -Force third_party | Out-Null
    $refs = [ordered]@{
        RAHFL = @('FangXiuwen/RAHFL', 'ba21203ca200ecdaa765539ba6aeacd05d2cbc99')
        AugHFL = @('FangXiuwen/AugHFL', 'ac4c64fbef0c6ccefae3bf796781e5097b590437')
        fl_data_quality = @('fardatalab/fl_data_quality', 'c9a5d584ce0805a5fcda70707f8ca640aabfc9a8')
    }
    $options = @('-c', 'http.lowSpeedLimit=1000', '-c', 'http.lowSpeedTime=20')
    if ($Proxy) { $options += @('-c', "http.proxy=$Proxy") }
    foreach ($name in $refs.Keys) {
        $repository, $commit = $refs[$name]
        $directory = "third_party/$name"
        if (-not (Test-Path -LiteralPath $directory)) {
            git init -q $directory
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            git -C $directory remote add origin "https://github.com/$repository.git"
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            git @options -C $directory fetch --depth 1 origin $commit
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            git -C $directory checkout --quiet --detach FETCH_HEAD
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
        $actual = git -C $directory rev-parse HEAD
        if ($LASTEXITCODE -ne 0 -or $actual -ne $commit) {
            throw "Expected $name at $commit, found $actual. Existing references are read-only."
        }
        Write-Output "$name $actual"
    }
} finally {
    Pop-Location
}
