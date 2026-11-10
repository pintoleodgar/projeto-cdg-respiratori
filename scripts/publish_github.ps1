<#
Publica o projeto no GitHub como repositório PRIVADO (o ICBHI tem licença própria
e os .docx com dados pessoais ficam de fora via .gitignore).

Pré-requisitos (uma vez):
    winget install GitHub.cli
    gh auth login

Uso:
    .\scripts\publish_github.ps1                  # cria <o teu utilizador>/projeto-cdg-respiratorio
    .\scripts\publish_github.ps1 -Nome outro-nome
    .\scripts\publish_github.ps1 -Publico         # só depois de rever licença e dados
#>
param(
    [string]$Nome = "projeto-cdg-respiratorio",
    [switch]$Publico
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

gh auth status | Out-Null
if (git status --porcelain) {
    git add -A
    git commit -m "Atualização antes de publicar no GitHub"
}
# Garante que nenhum .docx ou dataset vai no commit
$proibidos = git ls-files | Where-Object { $_ -match '\.docx$|^data/raw/|^_to_delete/' }
if ($proibidos) { throw "Ficheiros que não podem ser publicados: $($proibidos -join ', ')" }
# Nenhum nome de pessoa nos ficheiros versionados (anonimização do repositório)
$nomes = git grep -l -i -E "leodgar|rodrigo pereira|kravchenko|5004[0-9]{4}" -- . ":!*.pptx" 2>$null
if ($nomes) { throw "Ficheiros com nomes ou números de aluno: $($nomes -join ', ')" }

$vis = if ($Publico) { "--public" } else { "--private" }
if (-not (git remote | Select-String -Quiet '^origin$')) {
    gh repo create $Nome $vis --source . --remote origin --push `
        --description "TFC Ciência de Dados e Gestão — assistente de diagnóstico respiratório (ICBHI 2017)"
} else {
    git push -u origin main
}
gh repo view --web
