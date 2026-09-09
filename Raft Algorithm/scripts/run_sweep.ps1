$Episodes = @(5000, 10000)
$Nodes = @(25, 50, 75, 100)

Write-Host "Starting LLM MAPPO Parameter Sweep..."
foreach ($E in $Episodes) {
    foreach ($N in $Nodes) {
        $OutputFile = "runs/policies/llm_mappo_ep${E}_n${N}.json"
        if (Test-Path $OutputFile) {
            Write-Host "Skipping $OutputFile (already exists)"
            continue
        }
        Write-Host ""
        Write-Host "========================================================="
        Write-Host "Running $E episodes on $N nodes -> $OutputFile"
        Write-Host "========================================================="
        python -m rlraft.cli train --algorithm llm_mappo --episodes $E --nodes $N --output $OutputFile --require-llm
        if ($LASTEXITCODE -ne 0) {
            Write-Host "WARNING: Training failed for ep=${E} n=${N}"
        }
    }
}
Write-Host "Sweep complete."
