# PowerShell script to update backend URL after Render deployment
param(
    [Parameter(Mandatory=$true)]
    [string]$BackendUrl
)

Write-Host "Updating backend URL to: $BackendUrl" -ForegroundColor Green

# Update next.config.js
$nextConfigPath = "next.config.js"
$nextConfigContent = Get-Content $nextConfigPath -Raw

# Replace the backend URL in rewrites
$updatedContent = $nextConfigContent -replace 'http://localhost:8000', $BackendUrl

Set-Content -Path $nextConfigPath -Value $updatedContent

Write-Host "✅ Updated next.config.js" -ForegroundColor Green

# Update CORS settings in backend
$corsConfigPath = "backend/app/core/config.py"
$corsContent = Get-Content $corsConfigPath -Raw

# Add the new backend URL to allowed origins
$corsContent = $corsContent -replace '"https://\*\.render\.com"', "`"$BackendUrl`", `"https://*.render.com`""

Set-Content -Path $corsConfigPath -Value $corsContent

Write-Host "✅ Updated CORS settings" -ForegroundColor Green

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Commit changes: git add . && git commit -m 'Update backend URL'" -ForegroundColor Yellow
Write-Host "2. Push to GitHub: git push" -ForegroundColor Yellow
Write-Host "3. Redeploy frontend: vercel --prod" -ForegroundColor Yellow