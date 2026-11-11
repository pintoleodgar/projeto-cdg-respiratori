<#
Publica o projeto no GitHub como repositório PRIVADO (o ICBHI tem licença própria
e os .docx com dados pessoais ficam de fora via .gitignore).

Pré-requisitos (uma vez):
    winget install GitHub.cli
    gh auth login

Uso:
    .\scripts\publish_github.ps1                  # cria pintoleodgar/projeto-cdg-respiratori
    .\scripts\publish_github.ps1 -Nome outro-nome
    .\scripts\publish_github.ps1 -Publico         # só depois de rever licença e dados
#>
param(
    [string]$Nome = "projeto-cdg-respiratori",
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
# (o nome de utilizador do GitHub no badge é permitido; os CSV ficam fora para evitar falsos positivos em floats)
$nomes = git grep -l -i -E "\bleodgar garcia\b|\brodrigo pereira\b|kravchenko|\b5004[0-9]{4}\b" -- . ":!*.pptx" ":!*.csv" ":!scripts/publish_github.ps1" 2>$null
if ($nomes) { throw "Ficheiros com nomes ou números de aluno: $($nomes -join ', ')" }

$vis = if ($Publico) { "--public" } else { "--private" }
if (-not (git remote | Select-String -Quiet '^origin$')) {
    gh repo create $Nome $vis --source . --remote origin --push `
        --description "Protótipo de apoio ao diagnóstico respiratório a partir de sons de auscultação (ICBHI 2017)"
} else {
    git push -u origin main
}
gh repo view --web
