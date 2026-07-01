# =============================================================================
# run_golden_dataset.ps1 — Avaliacao das 40 perguntas no papel do Agente A6
# =============================================================================
# Avalia a RAG no papel do Agente A6 (Apoio Regulatorio e Licenciamento):
#   - injeta a persona do A6 no system prompt da geracao (--system-prompt)
#   - mede sinais proprios do A6: presenca de citacao e abstencao em baixa
#     confianca (a salvaguarda de mitigacao de alucinacoes)
#   - regista, por pergunta, o agente que a consultaria no servico transversal
# =============================================================================
param(
    [string]   $Model         = "qwen2.5:3b-instruct-q4_K_M",
    [string[]] $OnlySections  = @(),
    [int]      $SleepBetween  = 5,
    [string]   $A6PromptPath  = "evals\a6_system_prompt.txt",
    [switch]   $NoIndividual,                               # so o MD consolidado
    [switch]   $NoAgentPrompt,                              # correr sem a persona do A6
    [switch]   $NoTxt                                       # nao escrever o relatorio .txt
)
$Model = $Model.Trim()   # defensivo: um espaco a mais quebra o Ollama model lookup
# ---- Forcar UTF-8 no stdout/stderr do Python em Windows -------------------
# Sem isto, caracteres como o til combinante (U+0303), >= (U+2265) ou o grau
# (U+00B0) rebentam o print() do Python com UnicodeEncodeError no PowerShell.
# Cerca de 22% das respostas perderam-se por isto em corridas anteriores.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8       = "1"
# ---- Ativar/desativar o relatorio em texto simples ------------------------
$writeTxt = -not $NoTxt
# ---- Verificacoes de sanidade ---------------------------------------------
$venvPython = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "[error] $venvPython not found. Activate the venv first." `
        -ForegroundColor Red
    exit 1
}
try {
    $ollamaUp = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" `
        -TimeoutSec 5 -ErrorAction Stop
    Write-Host "[ok] Ollama responding, $($ollamaUp.models.Count) models loaded."
} catch {
    Write-Host "[error] Ollama not reachable on :11434. Start it first." `
        -ForegroundColor Red
    exit 1
}
try {
    $qdrant = Invoke-RestMethod -Uri "http://localhost:6333/collections/kb" `
        -TimeoutSec 5 -ErrorAction Stop
    Write-Host "[ok] Qdrant kb collection: $($qdrant.result.points_count) points."
} catch {
    Write-Host "[error] Qdrant not reachable on :6333." -ForegroundColor Red
    exit 1
}
# ---- Persona do Agente A6 -------------------------------------------------
# Escrita para ficheiro se ainda nao existir; podes edita-la a vontade depois.
# Reflete o papel do A6 no capitulo: corpus duplo (legislacao publica + dossie
# do projeto), citacao obrigatoria, distincao PT/UE e abstencao fundamentada.
$null = New-Item -ItemType Directory -Force -Path (Split-Path $A6PromptPath)
if (-not (Test-Path $A6PromptPath)) {
    $a6Prompt = @'
Es o Agente A6 - Apoio Regulatorio e Licenciamento, integrado num sistema
multiagente de apoio ao projeto de parques eolicos. A tua funcao e instruir e
verificar materia regulatoria e de licenciamento com base no corpus disponivel:
legislacao nacional e europeia, normas tecnicas (IEC, IEEE, EN), manuais dos
operadores de rede (REN, E-REDES) e o dossie tecnico do projeto.
Regras de funcionamento:
1. Responde exclusivamente com base nas passagens recuperadas. Nao inventes
   requisitos, valores, prazos ou referencias que nao constem do contexto.
2. Fundamenta cada afirmacao relevante na fonte especifica: diploma e artigo ou
   numero (por exemplo, Portaria n.o 73/2020, art. X), norma e clausula (por
   exemplo, IEC 60287-1-1), ou o documento do dossie do projeto.
3. Quando a pergunta o justificar, identifica a fase de licenciamento aplicavel
   e a documentacao exigida, e sinaliza omissoes ou nao-conformidades.
4. Distingue o que e requisito nacional (Portugal) do que decorre de regulamento
   europeu.
5. Se as passagens recuperadas nao sustentarem uma resposta, declara-o de forma
   explicita, em vez de especular. E preferivel assumir a ausencia de fundamento
   a produzir uma resposta plausivel mas nao verificavel.
6. Escreve em portugues europeu, com registo tecnico e objetivo.
Estrutura a resposta em duas partes: primeiro a resposta fundamentada; depois,
sob o titulo "Fontes", a lista das referencias efetivamente utilizadas.
'@
    $a6Prompt | Out-File -Encoding utf8 $A6PromptPath
    Write-Host "[a6] system prompt criado em $A6PromptPath"
}
# ---- Deteta suporte a --system-prompt no query.py -------------------------
$useSystemPrompt = $false
if (-not $NoAgentPrompt) {
    $helpText = (& $venvPython "scripts\query.py" --help 2>&1 | Out-String)
    if ($helpText -match 'system-prompt') {
        $useSystemPrompt = $true
        Write-Host "[a6] persona ATIVA (query.py aceita --system-prompt)." `
            -ForegroundColor Green
    } else {
        Write-Host "[a6][warn] query.py nao expoe --system-prompt." `
            -ForegroundColor Yellow
        Write-Host "[a6][warn] A correr SEM a persona do A6 (resultado = RAG crua)." `
            -ForegroundColor Yellow
        Write-Host "[a6][warn] Adiciona o hook indicado no cabecalho para ativar." `
            -ForegroundColor Yellow
    }
} else {
    Write-Host "[a6] -NoAgentPrompt: a correr a RAG crua (baseline)." `
        -ForegroundColor Yellow
}
# ---- Seccao -> agente que consulta o servico transversal ------------------
# A6 responde sempre; esta coluna regista de que agente a consulta proviria no
# papel de servico transversal de consulta normativa (ver Tabela do capitulo).
$agentMap = @{
    "A" = "A6";  "B" = "A6";  "C" = "A6";  "D" = "A6"
    "E" = "A8";  "F" = "A6";  "G" = "A12"; "H" = "A12"
}
# ---- Heuristicas de avaliacao A6 ------------------------------------------
# cites_source: a resposta invoca uma fonte concreta? (presenca, nao exatidao)
$citePattern = 'DL\s*\d|Decreto-Lei|Portaria|Despacho|Diretiva|Regulamento|' +
               'IEC\s*\d|IEEE\s*\d|EN\s*\d|ISO\s*\d|RSLEAT|RSSPTS|RSRDEEBT|RfG|' +
               'artigo|art\.|n\.o|n\.º|clausula|cláusula|anexo'
# abstained: a resposta assume ausencia de fundamento?
$abstainPattern = 'nao foi|nao foram|nao e possivel|nao consta|nao disponho|' +
                  'nao ha|corpus nao|sem fundamento|nao recuperad|nao encontrad|' +
                  'ausencia de fundamento|nao permite|' +
                  'não foi|não foram|não é possível|não consta|não há|sem fundamento'
# ---- As 40 perguntas, agrupadas por seccao (materias regulatorias do A6) ---
# Cada seccao e uma materia regulatoria do ambito do agente A6 (Apoio
# Regulatorio e Licenciamento), alinhada com o catalogo do Cap. 2 e o
# faseamento do Cap. 4. Todas tem resposta na legislacao/normas publicas
# indexadas na colecao kb (nao dependem do dossie confidencial):
#   A Classificacao e enquadramento (Tipo D, RfG, Despacho 7/2018, Portaria 73/2020)
#   B Acesso e reserva de capacidade (TRC, DL 15/2022 e 99/2024, RARI / Diretiva 3/2025)
#   C Licenciamento de producao (licenca de producao, DGEG, fases)
#   D Avaliacao de impacte ambiental (AIA, DIA, TUA, APA)
#   E Requisitos tecnicos de ligacao (FRT, capacidade reativa, conformidade RfG da REN)
#   F Documentacao e instrucao do dossie (Termos de Responsabilidade, Projeto de Execucao)
#   G Comunicacao operacional e exploracao (ensaios da REN, IEC 61400-21 / 61400-26)
#   H Hibrida e ciberseguranca (ponto de entrega partilhado, NIS2, IEC 62443)
$questions = @(
    @("A","tipo_D_limiar_tensao",
      "A partir de que nivel de tensao de ligacao e um modulo de parque gerador classificado como Tipo D em Portugal continental?"),
    @("A","tipo_D_limiar_potencia",
      "A partir de que potencia maxima injetada e um modulo gerador classificado como Tipo D, independentemente do nivel de tensao?"),
    @("A","rfg_requisitos_ligacao",
      "Que regulamento europeu estabelece os requisitos de ligacao a rede dos modulos geradores e por que sigla e conhecido?"),
    @("A","portaria73_parametrizacao_rfg",
      "Que diploma nacional parametriza para Portugal os requisitos de ligacao do RfG aplicaveis aos modulos geradores?"),
    @("A","categorias_tipo_A_a_D",
      "Como se distinguem as categorias de modulos geradores do Tipo A ao Tipo D no codigo de rede europeu?"),
    @("B","titulo_reserva_capacidade",
      "Que titulo habilita a reserva de capacidade de injecao na RESP e ao abrigo de que regime e emitido?"),
    @("B","dl15_2022_regime_renovaveis",
      "Que decreto-lei estabelece o regime juridico da producao de eletricidade a partir de fontes renovaveis e qual a sua revisao mais recente?"),
    @("B","rari_acesso_com_restricoes",
      "Em que consiste o acesso a rede com restricoes (RARI) e que entidade aprova as suas condicoes gerais?"),
    @("B","diretiva3_2025_restricoes_injecao",
      "Que instrumento regulatorio permite a injecao de potencia ativa sujeita a restricoes temporarias e que entidade o aprovou?"),
    @("B","entidade_competente_reserva",
      "Que entidade e competente para a atribuicao do Titulo de Reserva de Capacidade no quadro do regime de acesso a rede?"),
    @("C","licenca_producao_ato_entidade",
      "Que ato habilita a exploracao de um centro electroprodutor e que entidade o emite?"),
    @("C","fases_licenciamento_producao",
      "Quais as principais fases do licenciamento de um centro electroprodutor renovavel, do titulo de reserva a exploracao?"),
    @("C","modalidades_controlo_previo",
      "Que modalidades de controlo previo (licenca, comunicacao previa) estao previstas para os centros electroprodutores e de que dependem?"),
    @("C","competencias_dgeg_licenciamento",
      "Que competencias detem a DGEG no licenciamento de centros electroprodutores?"),
    @("C","pecas_tecnicas_pedido_licenca",
      "Que pecas tecnicas instruem o pedido de licenca de producao de um parque eolico?"),
    @("D","aia_condicoes_sujeicao",
      "Em que condicoes um parque eolico fica sujeito a procedimento de Avaliacao de Impacte Ambiental em Portugal?"),
    @("D","dia_natureza_efeitos",
      "O que e a Declaracao de Impacte Ambiental (DIA) e que efeito tem sobre o licenciamento subsequente?"),
    @("D","tua_titulo_unico_ambiental",
      "O que e o Titulo Unico Ambiental (TUA) e que regime o estabelece?"),
    @("D","apa_coordenacao_aia",
      "Que entidade coordena o procedimento de AIA e a emissao da DIA?"),
    @("D","regime_juridico_aia",
      "Que diploma estabelece o regime juridico da Avaliacao de Impacte Ambiental aplicavel a parques eolicos?"),
    @("E","frt_suportabilidade_cavas",
      "Que requisito de suportabilidade a cavas de tensao (Fault Ride Through) se aplica aos modulos Tipo D e onde esta parametrizado?"),
    @("E","capacidade_reativa_curvas_PQ",
      "Que exigencias de capacidade de potencia reativa (curvas P-Q) se aplicam aos modulos geradores Tipo D em Portugal?"),
    @("E","injecao_corrente_reativa_defeito",
      "Que requisito de injecao rapida de corrente reativa durante defeitos se aplica aos modulos Tipo D?"),
    @("E","gamas_frequencia_tensao_operacao",
      "Quais as gamas de frequencia e de tensao em que um modulo Tipo D tem de permanecer ligado a rede?"),
    @("E","conformidade_rfg_ensaios_ren",
      "Que processo de verificacao de conformidade com o RfG e conduzido pela REN para um novo modulo gerador?"),
    @("F","termo_responsabilidade_funcao",
      "Qual a funcao de um Termo de Responsabilidade no dossie de licenciamento de um parque eolico?"),
    @("F","projeto_execucao_conteudo",
      "Que elementos deve conter o Projeto de Execucao das instalacoes eletricas de um parque eolico?"),
    @("F","documentacao_licenciamento_eletrico",
      "Que documentacao tecnica e exigida na fase de licenciamento eletrico de um parque eolico em Portugal?"),
    @("F","pedido_elementos_adicionais",
      "O que motiva tipicamente um pedido de elementos adicionais pelas entidades licenciadoras e como se evita?"),
    @("F","esquema_unifilar_pecas_desenhadas",
      "Que pecas desenhadas, como o esquema unifilar, integram o projeto eletrico submetido a licenciamento?"),
    @("G","comunicacao_operacional_provisoria",
      "O que e a comunicacao operacional provisoria e que ensaios a antecedem?"),
    @("G","ensaios_conformidade_ren",
      "Que ensaios de conformidade conduz a REN antes da exploracao comercial de um parque eolico?"),
    @("G","iec61400_21_ensaios_qualidade",
      "Que norma define os ensaios de caracterizacao da qualidade de energia de aerogeradores e o que abrange?"),
    @("G","comunicacao_operacional_definitiva",
      "Que condicoes habilitam a passagem da comunicacao operacional provisoria a definitiva?"),
    @("G","indicadores_disponibilidade_61400_26",
      "Que norma estabelece o modelo normalizado de indicadores de disponibilidade de parques eolicos?"),
    @("H","hibrida_ponto_entrega_partilhado",
      "Como e tratado o licenciamento de uma central hibrida que partilha ponto de entrega entre producao eolica e fotovoltaica?"),
    @("H","nis2_transposicao_pt",
      "Que diploma transpoe para Portugal a Diretiva NIS2 e a que operadores se aplica?"),
    @("H","iec62443_segmentacao_zonas",
      "Que norma estabelece a segmentacao por zonas e condutas para a ciberseguranca de sistemas de automacao industrial?"),
    @("H","isolamento_rede_ot_scada",
      "Que principios de isolamento se aplicam a interligacao de sistemas de IA a rede operacional (OT) e ao SCADA de um parque?"),
    @("H","obrigacoes_ciberseguranca_nis2",
      "Que obrigacoes de ciberseguranca recaem sobre os operadores de infraestruturas energeticas criticas ao abrigo da NIS2?")
)
# ---- Filtrar por seccao, se pedido ----------------------------------------
if ($OnlySections.Count -gt 0) {
    $upper = $OnlySections | ForEach-Object { $_.ToUpper() }
    $questions = $questions | Where-Object { $upper -contains $_[0] }
    Write-Host "[filter] running $($questions.Count) questions from sections: $($upper -join ',')"
}
# ---- Pasta de saida -------------------------------------------------------
$stamp  = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = "evals\a6_$stamp"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$consolidated = Join-Path $outDir "all_answers.md"
$personaState = if ($useSystemPrompt) { "ON" } else { "OFF" }
Write-Host "[output]       $outDir"
Write-Host "[consolidated] $consolidated"
Write-Host "[model]        $Model"
Write-Host "[a6 persona]   $personaState"
Write-Host "[count]        $($questions.Count) questions"
if ($NoIndividual) {
    Write-Host "[mode]         consolidated MD only (no per-question .txt)"
}
Write-Host ""
# ---- Escrever o cabecalho do MD consolidado -------------------------------
$sectionList = if ($OnlySections.Count -gt 0) { $OnlySections -join ', ' } else { 'all' }
$startedAt   = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$mdHeader = @"
# Agente A6 - Apoio Regulatorio e Licenciamento - Corrida $stamp
- Model: ``$Model``
- Persona A6 (system prompt): **$personaState**
- Sections: $sectionList
- Questions: $($questions.Count)
- Started: $startedAt
Avaliacao da RAG no papel do agente A6. Para alem do retrieval/rerank, regista,
por pergunta: se a resposta invoca uma fonte concreta (``cites_source``), se
assume ausencia de fundamento (``abstained``) e de que agente proviria a
consulta no servico transversal (``consulting_agent``).
Este documento e escrito apos cada pergunta, pelo que uma corrida que falhe a
meio mantem aqui tudo o que ja completou.
---
"@
$mdHeader | Out-File -Encoding utf8 $consolidated
# ---- Cabecalho do manifest ------------------------------------------------
$manifest = Join-Path $outDir "manifest.csv"
"idx,section,consulting_agent,label,elapsed_s,tier,top_rerank,cites_source,abstained,filter_applied,source_types,jurisdictions,output_file" `
    | Out-File -Encoding utf8 $manifest
# ---- Relatorio em texto simples: cabecalho (escrito uma vez) --------------
$txtReport = Join-Path $outDir "a6_report.txt"
if ($writeTxt) {
    $hdr = New-Object System.Text.StringBuilder
    [void]$hdr.AppendLine("================================================================================")
    [void]$hdr.AppendLine("Avaliacao da RAG no papel do Agente A6 - Apoio Regulatorio e Licenciamento")
    [void]$hdr.AppendLine("================================================================================")
    [void]$hdr.AppendLine("Corrida   : $stamp")
    [void]$hdr.AppendLine("Modelo    : $Model")
    [void]$hdr.AppendLine("Persona A6 (system prompt): $personaState")
    [void]$hdr.AppendLine("Seccoes   : $sectionList")
    [void]$hdr.AppendLine("Perguntas : $($questions.Count)")
    [void]$hdr.AppendLine("Inicio    : $startedAt")
    [void]$hdr.AppendLine("")
    [void]$hdr.AppendLine("Por pergunta regista-se a resposta, as fontes recuperadas e os sinais do A6")
    [void]$hdr.AppendLine("(cites_source, abstained). Os sinais sao heuristicos e devem ser confirmados")
    [void]$hdr.AppendLine("em revisao humana contra as fontes recuperadas em cada bloco.")
    [void]$hdr.AppendLine("")
    $hdr.ToString() | Out-File -Encoding utf8 $txtReport
    Write-Host "[txt]          $txtReport"
}
# ---- Auxiliar: extrair seccoes do output do query.py ----------------------
# O query.py imprime numa estrutura previsivel. Isolamos quatro coisas:
#   1. O bloco "Top N candidates" (ja bem alinhado)
#   2. O bloco "Final sources after reranker"
#   3. A linha "Retrieval confidence tier"
#   4. A propria resposta (entre ">> Answer (...):" e ">> Retrieval summary:")
function Get-OutputSections {
    param([string]$Text)
    $top12 = ""
    $final = ""
    $answer = ""
    $tier = ""
    if ($Text -match '(?ms)^>> Top \d+ candidates entering reranker:\s*\r?\n(.+?)\r?\n\s*\r?\n>> Reranking') {
        $top12 = $matches[1].Trim()
    }
    if ($Text -match '(?ms)^>> Final sources after reranker:\s*\r?\n(.+?)\r?\n\s*\r?\n>>') {
        $final = $matches[1].Trim()
    }
    if ($Text -match '>> Retrieval confidence tier:\s*(\w+)\s*\(top rerank = ([\d\.]+)\)') {
        $tier = "$($matches[1]) (rerank=$($matches[2]))"
    }
    if ($Text -match '(?ms)>> Answer \([^)]*\):\s*\r?\n(.+?)\r?\n\s*\r?\n>> Retrieval summary:') {
        $answer = $matches[1].Trim()
    } elseif ($Text -match '(?ms)>> Answer \([^)]*\):\s*\r?\n(.+)$') {
        $answer = $matches[1].Trim()
    }
    return @{
        Top12  = $top12
        Final  = $final
        Tier   = $tier
        Answer = $answer
    }
}
# ---- Ciclo principal ------------------------------------------------------
$totalStart = Get-Date
$idx = 0
foreach ($q in $questions) {
    $idx++
    $section = $q[0]
    $label   = $q[1]
    $text    = $q[2]
    $idxStr  = "{0:D2}" -f $idx
    $consult = $agentMap[$section]
    $outFile = Join-Path $outDir "Q${idxStr}_${section}_${label}.txt"
    Write-Host "[$idxStr/$($questions.Count)] [$section -> $consult] $label" `
        -ForegroundColor Cyan
    Write-Host "    Q: $text"
    # Construir a invocacao do query.py (persona via --system-prompt, se suportado)
    $qpArgs = @("scripts\query.py", $text, "--model", $Model)
    if ($useSystemPrompt) { $qpArgs += @("--system-prompt", $A6PromptPath) }
    $qStart = Get-Date
    $output = & $venvPython @qpArgs 2>&1
    $qEnd   = Get-Date
    $elapsed = [math]::Round(($qEnd - $qStart).TotalSeconds, 1)
    $outText = $output -join "`n"
    # .txt por pergunta (a menos que -NoIndividual)
    if (-not $NoIndividual) {
        "QUESTION: $text"                       | Out-File -Encoding utf8 $outFile
        "AGENT: A6 (Apoio Regulatorio)"         | Out-File -Encoding utf8 -Append $outFile
        "CONSULTING_AGENT: $consult"            | Out-File -Encoding utf8 -Append $outFile
        "MODEL: $Model"                         | Out-File -Encoding utf8 -Append $outFile
        "A6_PERSONA: $personaState"             | Out-File -Encoding utf8 -Append $outFile
        "ELAPSED_S: $elapsed"                   | Out-File -Encoding utf8 -Append $outFile
        "TIMESTAMP: $(Get-Date -Format 'o')"    | Out-File -Encoding utf8 -Append $outFile
        ""                                      | Out-File -Encoding utf8 -Append $outFile
        $output                                 | Out-File -Encoding utf8 -Append $outFile
    }
    # Extrair seccoes estruturadas
    $parts = Get-OutputSections $outText
    # Metricas do manifest
    $filter = if ($outText -match 'filter applied:\s*(.+)') { $matches[1].Trim() } else { "" }
    $topRk  = if ($outText -match 'top rerank score:\s*([\d\.]+)') { $matches[1] } else { "" }
    $types  = if ($outText -match 'source types:\s*(.+)') { $matches[1].Trim() } else { "" }
    $juris  = if ($outText -match 'jurisdictions:\s*(.+)') { $matches[1].Trim() } else { "" }
    # Sinais especificos do A6 (heuristicas sobre o texto da resposta)
    $cites    = if ($parts.Answer -and ($parts.Answer -match $citePattern))    { "yes" } else { "no" }
    $abstain  = if ($parts.Answer -and ($parts.Answer -match $abstainPattern)) { "yes" } else { "no" }
    # Seguro para CSV (virgulas dentro de campos -> ponto-e-virgula)
    $filterCsv = $filter -replace ',', ';'
    $typesCsv  = $types  -replace ',', ';'
    $jurisCsv  = $juris  -replace ',', ';'
    $tierCsv   = ($parts.Tier -replace ',', ';')
    "$idx,$section,$consult,$label,$elapsed,$tierCsv,$topRk,$cites,$abstain,$filterCsv,$typesCsv,$jurisCsv,$(Split-Path -Leaf $outFile)" `
        | Out-File -Encoding utf8 -Append $manifest
    # Construir o bloco Markdown desta pergunta
    $md = New-Object System.Text.StringBuilder
    [void]$md.AppendLine("## Q$idxStr - [$section] $label")
    [void]$md.AppendLine("")
    [void]$md.AppendLine("**Question (PT):** $text")
    [void]$md.AppendLine("")
    [void]$md.AppendLine("| Field            | Value |")
    [void]$md.AppendLine("|------------------|-------|")
    [void]$md.AppendLine("| Consulting agent | $consult |")
    [void]$md.AppendLine("| Tier             | $($parts.Tier) |")
    [void]$md.AppendLine("| Top rerank       | $topRk |")
    [void]$md.AppendLine("| Cites source     | $cites |")
    [void]$md.AppendLine("| Abstained        | $abstain |")
    [void]$md.AppendLine("| Filter applied   | $filter |")
    [void]$md.AppendLine("| Source types     | $types |")
    [void]$md.AppendLine("| Jurisdictions    | $juris |")
    [void]$md.AppendLine("| Elapsed          | $elapsed s |")
    [void]$md.AppendLine("| Timestamp        | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') |")
    [void]$md.AppendLine("")
    if ($parts.Top12) {
        [void]$md.AppendLine("**Top candidates (pre-rerank):**")
        [void]$md.AppendLine("")
        [void]$md.AppendLine('```')
        [void]$md.AppendLine($parts.Top12)
        [void]$md.AppendLine('```')
        [void]$md.AppendLine("")
    }
    if ($parts.Final) {
        [void]$md.AppendLine("**Final sources after reranker (top-5):**")
        [void]$md.AppendLine("")
        [void]$md.AppendLine('```')
        [void]$md.AppendLine($parts.Final)
        [void]$md.AppendLine('```')
        [void]$md.AppendLine("")
    }
    [void]$md.AppendLine("**Answer (A6):**")
    [void]$md.AppendLine("")
    if ($parts.Answer) {
        # Indentar a resposta como citacao para a destacar visualmente.
        $answerLines = $parts.Answer -split "`n"
        foreach ($line in $answerLines) {
            [void]$md.AppendLine("> $line")
        }
    } else {
        [void]$md.AppendLine("> _(no answer recovered - likely crashed before generation)_")
    }
    [void]$md.AppendLine("")
    [void]$md.AppendLine("---")
    [void]$md.AppendLine("")
    Add-Content -Path $consolidated -Value $md.ToString() -Encoding utf8
    # ---- Bloco em texto simples desta pergunta (incremental, resistente a falhas) ----
    if ($writeTxt) {
        $tb = New-Object System.Text.StringBuilder
        [void]$tb.AppendLine("================================================================================")
        [void]$tb.AppendLine("Q$idxStr  [$section]  $label")
        [void]$tb.AppendLine("================================================================================")
        [void]$tb.AppendLine("Pergunta: $text")
        [void]$tb.AppendLine("")
        [void]$tb.AppendLine("Consulting agent : $consult")
        [void]$tb.AppendLine("Tier             : $($parts.Tier)")
        [void]$tb.AppendLine("Top rerank       : $topRk")
        [void]$tb.AppendLine("Cites source     : $cites")
        [void]$tb.AppendLine("Abstained        : $abstain")
        [void]$tb.AppendLine("Filter applied   : $filter")
        [void]$tb.AppendLine("Source types     : $types")
        [void]$tb.AppendLine("Jurisdictions    : $juris")
        [void]$tb.AppendLine("Elapsed          : $elapsed s")
        [void]$tb.AppendLine("")
        if ($parts.Top12) {
            [void]$tb.AppendLine("--- Top candidates (pre-rerank) ---")
            [void]$tb.AppendLine($parts.Top12)
            [void]$tb.AppendLine("")
        }
        if ($parts.Final) {
            [void]$tb.AppendLine("--- Final sources after reranker (top-5) ---")
            [void]$tb.AppendLine($parts.Final)
            [void]$tb.AppendLine("")
        }
        [void]$tb.AppendLine("--- Resposta (A6) ---")
        if ($parts.Answer) {
            [void]$tb.AppendLine($parts.Answer)
        } else {
            [void]$tb.AppendLine("(no answer recovered - likely crashed before generation)")
        }
        [void]$tb.AppendLine("")
        Add-Content -Path $txtReport -Value $tb.ToString() -Encoding utf8
    }
    Write-Host "    -> $elapsed s, tier=$($parts.Tier), cites=$cites, abstained=$abstain" `
        -ForegroundColor Green
    # Verificacao rapida de saude do Ollama entre perguntas
    try {
        Invoke-RestMethod -Uri "http://localhost:11434/api/tags" `
            -TimeoutSec 5 -ErrorAction Stop | Out-Null
    } catch {
        Write-Host "[warn] Ollama unresponsive after Q$idxStr. Sleeping 30s..." `
            -ForegroundColor Yellow
        Start-Sleep -Seconds 30
    }
    Start-Sleep -Seconds $SleepBetween
}
# ---- Rodape no MD consolidado ---------------------------------------------
$totalEnd = Get-Date
$totalMin = [math]::Round(($totalEnd - $totalStart).TotalMinutes, 1)
$finishedAt = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
# Resumo agregado dos sinais A6 (lido do manifest)
$rows = Import-Csv $manifest
$nCite    = ($rows | Where-Object { $_.cites_source -eq 'yes' }).Count
$nAbstain = ($rows | Where-Object { $_.abstained   -eq 'yes' }).Count
$nTotal   = $rows.Count
$mdFooter = @"
---
**Run complete.** $nTotal questions in $totalMin minutes.
Finished at $finishedAt.
## Sinais agregados do A6
- Respostas com citacao de fonte: $nCite / $nTotal
- Respostas com abstencao fundamentada: $nAbstain / $nTotal
- Persona A6 (system prompt): $personaState
> Nota: ``cites_source`` e ``abstained`` sao heuristicas sobre o texto (presenca
> de referencia / linguagem de abstencao), uteis para triagem. A exatidao das
> citacoes e a correcao da abstencao devem ser confirmadas na revisao humana,
> contra as fontes recuperadas registadas em cada bloco.
"@
Add-Content -Path $consolidated -Value $mdFooter -Encoding utf8
# ---- Relatorio em texto simples: agregado + fecho -------------------------
if ($writeTxt) {
    $ftr = New-Object System.Text.StringBuilder
    [void]$ftr.AppendLine("")
    [void]$ftr.AppendLine("================================================================================")
    [void]$ftr.AppendLine("Sinais agregados do A6")
    [void]$ftr.AppendLine("================================================================================")
    [void]$ftr.AppendLine("Questions               : $nTotal in $totalMin min")
    [void]$ftr.AppendLine("Respostas com citacao   : $nCite / $nTotal")
    [void]$ftr.AppendLine("Respostas com abstencao : $nAbstain / $nTotal")
    [void]$ftr.AppendLine("Persona A6              : $personaState")
    [void]$ftr.AppendLine("")
    [void]$ftr.AppendLine("Nota: cites_source e abstained sao heuristicas sobre o texto; a exatidao das")
    [void]$ftr.AppendLine("citacoes e a correcao da abstencao confirmam-se contra as fontes recuperadas.")
    Add-Content -Path $txtReport -Value $ftr.ToString() -Encoding utf8
}
Write-Host ""
Write-Host "[done] $nTotal questions in $totalMin minutes." `
    -ForegroundColor Green
Write-Host "[done] cites_source=$nCite/$nTotal  abstained=$nAbstain/$nTotal  persona=$personaState"
Write-Host "[done] Consolidated: $consolidated"
Write-Host "[done] Manifest:     $manifest"
if ($writeTxt) {
    Write-Host "[done] TXT report:   $txtReport"
}
if (-not $NoIndividual) {
    Write-Host "[done] Per-question files in $outDir"
}
