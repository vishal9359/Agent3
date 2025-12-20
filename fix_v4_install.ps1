# Fix Agent5 V4 Installation Script
# This script clears cache, pulls latest changes, and reinstalls the package

Write-Host "=" -ForegroundColor Cyan
Write-Host "Agent5 V4 - Installation Fix Script" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan
Write-Host ""

# Step 1: Pull latest changes
Write-Host "[1/6] Pulling latest changes from version4 branch..." -ForegroundColor Yellow
git fetch origin
git pull origin version4
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Git pull failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Latest changes pulled" -ForegroundColor Green
Write-Host ""

# Step 2: Uninstall existing package
Write-Host "[2/6] Uninstalling existing agent5 package..." -ForegroundColor Yellow
pip uninstall agent5 -y 2>$null
Write-Host "✓ Package uninstalled" -ForegroundColor Green
Write-Host ""

# Step 3: Clear Python cache
Write-Host "[3/6] Clearing Python cache..." -ForegroundColor Yellow
Get-ChildItem -Path . -Filter "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force
Get-ChildItem -Path . -Filter "*.pyc" -Recurse -File | Remove-Item -Force
Get-ChildItem -Path . -Filter "*.pyo" -Recurse -File | Remove-Item -Force
if (Test-Path "agent5.egg-info") {
    Remove-Item -Recurse -Force "agent5.egg-info"
}
if (Test-Path "build") {
    Remove-Item -Recurse -Force "build"
}
if (Test-Path "dist") {
    Remove-Item -Recurse -Force "dist"
}
Write-Host "✓ Cache cleared" -ForegroundColor Green
Write-Host ""

# Step 4: Verify wrapper functions exist
Write-Host "[4/6] Verifying wrapper functions..." -ForegroundColor Yellow
$functions = @(
    @{File="agent5\bottom_up_aggregator.py"; Function="aggregate_semantics"},
    @{File="agent5\leaf_semantic_extractor.py"; Function="extract_leaf_semantics"},
    @{File="agent5\sfm_builder.py"; Function="build_scenario_flow_model"},
    @{File="agent5\mermaid_translator.py"; Function="translate_to_mermaid"}
)

$allFound = $true
foreach ($func in $functions) {
    if (Select-String -Path $func.File -Pattern "^def $($func.Function)" -Quiet) {
        Write-Host "  ✓ $($func.Function) found in $($func.File)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $($func.Function) NOT found in $($func.File)" -ForegroundColor Red
        $allFound = $false
    }
}

if (-not $allFound) {
    Write-Host ""
    Write-Host "ERROR: Some wrapper functions are missing!" -ForegroundColor Red
    Write-Host "Please ensure you're on the version4 branch and have pulled the latest changes." -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 5: Install package in editable mode
Write-Host "[5/6] Installing agent5 in editable mode..." -ForegroundColor Yellow
pip install -e .
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: pip install failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Package installed" -ForegroundColor Green
Write-Host ""

# Step 6: Verify installation
Write-Host "[6/6] Verifying installation..." -ForegroundColor Yellow
$commands = @("agent5", "agent5-v4")
foreach ($cmd in $commands) {
    $result = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($result) {
        Write-Host "  ✓ $cmd command is available" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $cmd command NOT found" -ForegroundColor Red
    }
}
Write-Host ""

Write-Host "=" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan
Write-Host ""
Write-Host "Test the command with:" -ForegroundColor Cyan
Write-Host "  agent5-v4 --help" -ForegroundColor White
Write-Host ""




