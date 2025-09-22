
Write-Host "Step1: Conversion"

python SkeletonConverter.py

Write-Host "-------------------------"
Write-Host "Step2: Feature extraction"

python mmaction2/mmaction2/extractSkFeats.py

Write-Host "-------------------------"
Write-Host "Step3: Create split"

scripts/createSplitPipline.ps1
