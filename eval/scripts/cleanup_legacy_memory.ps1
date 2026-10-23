#requires -Version 7.0
[CmdletBinding()]
param([switch]$Apply)

$ErrorActionPreference = 'Stop'
$legacyBase = [IO.Path]::GetFullPath((Join-Path $env:USERPROFILE '.harness'))
$legacyProjects = Join-Path $legacyBase 'projects'
$legacyUser = Join-Path $legacyBase 'memory'
$legacyTargets = @()
if (Test-Path -LiteralPath $legacyUser) { $legacyTargets += $legacyUser }
if (Test-Path -LiteralPath $legacyProjects) {
    foreach ($projectDirectory in Get-ChildItem -LiteralPath $legacyProjects -Directory -Force) {
        if ($projectDirectory.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Project directory is a reparse point: $($projectDirectory.FullName)"
        }
        $candidateDirectory = Join-Path $projectDirectory.FullName 'memory-v2'
        if (Test-Path -LiteralPath $candidateDirectory) { $legacyTargets += $candidateDirectory }
    }
}

$cleanupPlan = @()
foreach ($targetDirectory in $legacyTargets) {
    $resolvedDirectory = (Resolve-Path -LiteralPath $targetDirectory).ProviderPath
    $allowedUser = [String]::Equals($resolvedDirectory, $legacyUser, [StringComparison]::OrdinalIgnoreCase)
    $relativeDirectory = [IO.Path]::GetRelativePath($legacyProjects, $resolvedDirectory)
    $allowedProject = $resolvedDirectory.StartsWith($legacyProjects + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase) -and $relativeDirectory -match '^[^\\]+\\memory-v2$'
    if (-not ($allowedUser -or $allowedProject)) { throw "Outside approved legacy roots: $resolvedDirectory" }

    $ancestorDirectory = Get-Item -LiteralPath $resolvedDirectory -Force
    while ($null -ne $ancestorDirectory -and $ancestorDirectory.FullName.Length -ge $legacyBase.Length) {
        if ($ancestorDirectory.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Reparse point: $($ancestorDirectory.FullName)"
        }
        $ancestorDirectory = $ancestorDirectory.Parent
    }
    $descendants = @(Get-ChildItem -LiteralPath $resolvedDirectory -Force -Recurse)
    if (@($descendants | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
        throw "Reparse point inside $resolvedDirectory"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $resolvedDirectory 'memory.db'))) {
        throw "Expected legacy memory database missing: $resolvedDirectory"
    }
    $cleanupPlan += [ordered]@{
        path = $resolvedDirectory
        files = @($descendants | Where-Object { -not $_.PSIsContainer } | ForEach-Object {
            [ordered]@{ name = [IO.Path]::GetRelativePath($resolvedDirectory, $_.FullName); bytes = $_.Length }
        })
    }
}

$record = [ordered]@{
    date_utc = [DateTime]::UtcNow.ToString('o')
    mode = $(if ($Apply) { 'apply' } else { 'preview' })
    roots = $cleanupPlan
    preserved = @('Harness session journals', 'Codex memories', 'memory-v3', 'all other project data')
    completed = $false
}
$record | ConvertTo-Json -Depth 8 | Write-Output
if (-not $Apply) {
    Write-Output 'Preview only. Close Harness, check the paths, then run this script with -Apply to delete.'
    return
}

$activeHarness = @(Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and $_.Name -match '^(pythonw?|veriforge)(\.exe)?$' -and
    $_.CommandLine -match '(harness_code_agent|veriforge)' -and
    $_.CommandLine -notmatch '(pytest|eval[\\/]scripts)'
})
if ($activeHarness.Count) { throw 'Close running Harness processes before deleting legacy memory.' }

$resultsRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\results'))
New-Item -ItemType Directory -Path $resultsRoot -Force | Out-Null
$manifest = Join-Path $resultsRoot ('memory-legacy-cleanup-' + [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss-fff') + '.json')
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifest -Encoding utf8NoBOM
foreach ($entry in $cleanupPlan) {
    $current = (Resolve-Path -LiteralPath $entry.path).ProviderPath
    if (-not [String]::Equals($current, $entry.path, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target changed since preview: $($entry.path)"
    }
    if ((Get-Item -LiteralPath $current -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw "Target became a reparse point: $current"
    }
    Remove-Item -LiteralPath $current -Recurse -Force
    if (Test-Path -LiteralPath $current) { throw "Legacy cleanup incomplete: $current" }
}
$record.completed = $true
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifest -Encoding utf8NoBOM
Write-Output "Removed legacy roots: $($cleanupPlan.Count)"
Write-Output "Manifest: $manifest"
