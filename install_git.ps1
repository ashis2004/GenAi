# PowerShell Script to Install Git on Windows
# Run this script as Administrator for best results

Write-Host "🔧 Git Installation Helper" -ForegroundColor Green
Write-Host "=" * 50

# Method 1: Try Chocolatey (if available)
function Install-WithChocolatey {
    Write-Host "`n🍫 Checking for Chocolatey..." -ForegroundColor Yellow
    
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Host "✅ Chocolatey found! Installing Git..." -ForegroundColor Green
        try {
            choco install git -y
            Write-Host "✅ Git installed via Chocolatey!" -ForegroundColor Green
            return $true
        }
        catch {
            Write-Host "❌ Chocolatey installation failed: $_" -ForegroundColor Red
            return $false
        }
    }
    else {
        Write-Host "❌ Chocolatey not found" -ForegroundColor Red
        return $false
    }
}

# Method 2: Try Winget (Windows Package Manager)
function Install-WithWinget {
    Write-Host "`n📦 Checking for Winget..." -ForegroundColor Yellow
    
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "✅ Winget found! Installing Git..." -ForegroundColor Green
        try {
            winget install --id Git.Git -e --source winget
            Write-Host "✅ Git installed via Winget!" -ForegroundColor Green
            return $true
        }
        catch {
            Write-Host "❌ Winget installation failed: $_" -ForegroundColor Red
            return $false
        }
    }
    else {
        Write-Host "❌ Winget not found" -ForegroundColor Red
        return $false
    }
}

# Method 3: Download and install manually
function Install-Manually {
    Write-Host "`n🌐 Manual installation method..." -ForegroundColor Yellow
    
    $gitUrl = "https://github.com/git-for-windows/git/releases/latest/download/Git-2.45.2-64-bit.exe"
    $downloadPath = "$env:TEMP\GitInstaller.exe"
    
    try {
        Write-Host "📥 Downloading Git installer..." -ForegroundColor Green
        Invoke-WebRequest -Uri $gitUrl -OutFile $downloadPath -UseBasicParsing
        
        Write-Host "🚀 Running Git installer..." -ForegroundColor Green
        Start-Process -FilePath $downloadPath -ArgumentList "/SILENT" -Wait
        
        Write-Host "🧹 Cleaning up..." -ForegroundColor Green
        Remove-Item $downloadPath -Force
        
        Write-Host "✅ Git installation completed!" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ Manual installation failed: $_" -ForegroundColor Red
        return $false
    }
}

# Check if Git is already installed
if (Get-Command git -ErrorAction SilentlyContinue) {
    $version = git --version
    Write-Host "✅ Git is already installed: $version" -ForegroundColor Green
    exit 0
}

Write-Host "❌ Git not found. Attempting installation..." -ForegroundColor Red

# Try installation methods in order
$installed = $false

# Try Winget first (most modern)
if (-not $installed) {
    $installed = Install-WithWinget
}

# Try Chocolatey second
if (-not $installed) {
    $installed = Install-WithChocolatey
}

# Try manual download third
if (-not $installed) {
    Write-Host "`n⚠️  Package managers failed. Trying manual download..." -ForegroundColor Yellow
    $installed = Install-Manually
}

# Final check
if ($installed) {
    Write-Host "`n🎉 Installation completed!" -ForegroundColor Green
    Write-Host "🔄 Please close and reopen PowerShell, then run:" -ForegroundColor Yellow
    Write-Host "   git --version" -ForegroundColor White
    Write-Host "`n📝 Next steps:" -ForegroundColor Yellow
    Write-Host "   1. Close this PowerShell window" -ForegroundColor White
    Write-Host "   2. Open a new PowerShell window" -ForegroundColor White
    Write-Host "   3. Navigate to your project: cd C:\21MIP\LLM\GitHub\GenAi" -ForegroundColor White
    Write-Host "   4. Run: python github_manager.py" -ForegroundColor White
}
else {
    Write-Host "`n❌ All installation methods failed!" -ForegroundColor Red
    Write-Host "`n🔧 Manual Installation Required:" -ForegroundColor Yellow
    Write-Host "   1. Go to: https://git-scm.com/download/win" -ForegroundColor White
    Write-Host "   2. Download and run the installer" -ForegroundColor White
    Write-Host "   3. Use default settings during installation" -ForegroundColor White
    Write-Host "   4. Restart PowerShell after installation" -ForegroundColor White
}

Write-Host "`n" -ForegroundColor White
Read-Host "Press Enter to continue"
