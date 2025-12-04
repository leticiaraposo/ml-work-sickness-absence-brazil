
# Pacotes necessários --------------------------------------------------------
pkgs <- c(
  "readxl", "dplyr", "forcats", "rsample", "gtsummary",
  "flextable", "officer", "stringr", "purrr", "tidyr"
  # "cardx" # inclua se realmente usar
)

to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

invisible(lapply(pkgs, library, character.only = TRUE))

# Opções e reprodutibilidade -------------------------------------------------
set.seed(123)
options(stringsAsFactors = FALSE)

# Entrada (arquivo bruto) ----------------------------------------------------
dados <- readxl::read_excel("C:\\Users\\letic\\Meu Drive (raposo@peb.ufrj.br)\\UNIRIO\\Projetos\\2025\\TCC - Beatriz\\Dados\\dados_Beatriz_TCC_2025.xlsx")

# Pré-processamento ----------------------------------------------------------
dados_pp <- dados %>%
  # Remover variáveis não utilizadas
  dplyr::select(
    -NU_ANO, -NUTEMPORIS, -NUTEMPO, -TPTEMPORIS, -ID_OCUPA_N, -EVOLUCAO,
    -CONDUT_DES, -FUNCOES, -DIAG_ESP, -DIAG1, -nome_estado,
    -DT_NOTIFIC, -DT_DIAG, -TEMPO_DIAG_NOTIF, -CS_GESTANT,
    -AFAST_DESG, -INDIVIDUAL, -MUDA_TRAB, -COLETIVA, -CONDUTA, -NENHUM
  ) %>%
  # Filtragens (exclusões explícitas)
  dplyr::filter(
    !SIT_TRAB   %in% c("Desempregado","Aposentado","Outros","Não se aplica"),
    !TERCEIRIZA %in% c("Não se aplica"),
    !CAT        %in% c("Não se aplica")
  ) %>%
  # Recodificações de categorias
  dplyr::mutate(
    FUMA = dplyr::case_when(
      FUMA %in% c("Sim","Ex-fumante") ~ "Sim",
      TRUE ~ "Não"
    ),
    CS_ESCOL_N = dplyr::case_when(
      CS_ESCOL_N %in% c("Analfabeto","Ensino fundamental") ~ "Até ensino fundamental",
      TRUE ~ CS_ESCOL_N
    ),
    SIT_TRAB = dplyr::case_when(
      SIT_TRAB %in% c("Autônomo","Avulso","Empregado não registrado","Temporário",
                      "Cooperativado","Empregador") ~ "Outros",
      TRUE ~ SIT_TRAB
    ),
    CS_RACA = dplyr::case_when(
      CS_RACA %in% c("Preta","Parda") ~ "Preta/Parda",
      CS_RACA %in% c("Amarela","Indígena") ~ "Amarela/Indígena",
      TRUE ~ CS_RACA
    ),
    DIAG2 = dplyr::case_when(
      DIAG2 %in% c("Retardo mental","Transtornos mentais orgânicos",
                   "Transtorno mental não especificado") ~ "Transtorno mental não especificado",
      DIAG2 %in% c("Síndromes comportamentais associadas a perturbações fisiológicas e fatores físicos",
                   "Transtornos comportamentais e emocionais com início geralmente na infância e adolescência",
                   "Transtornos da personalidade e do comportamento em adultos",
                   "Transtornos mentais e comportamentais devido ao uso de substâncias psicoativas") ~
        "Transtornos de personalidade e comportamentais",
      TRUE ~ DIAG2
    ),
    # Faixas etárias (como categorias)
    NU_IDADE_N = dplyr::case_when(
      NU_IDADE_N >= 18 & NU_IDADE_N <= 29 ~ "18-29",
      NU_IDADE_N >= 30 & NU_IDADE_N <= 39 ~ "30-39",
      NU_IDADE_N >= 40 & NU_IDADE_N <= 49 ~ "40-49",
      NU_IDADE_N >= 50 & NU_IDADE_N <= 59 ~ "50-59",
      NU_IDADE_N >= 60 ~ "60 ou mais",
      TRUE ~ NA_character_
    )
  ) %>%
  # Transformar apenas colunas de texto em fator (evita corromper numéricos)
  dplyr::mutate(
    dplyr::across(where(is.character), forcats::as_factor)
  ) %>%
  # Padronizar nomes *de exibição* (mantendo objetos sintáticos para análise)
  dplyr::rename(
    faixa_etaria = NU_IDADE_N,
    sexo = CS_SEXO,
    escolaridade = CS_ESCOL_N,
    raca_cor = CS_RACA,
    terceirizado = TERCEIRIZA,
    regiao = regiao,
    alcool = ALCOOL,
    psicofarmacos = PSICO_FARM,
    drogas = DROGAS,
    fumante = FUMA,
    afastado_trabalho = AFAST_TRAB,
    situacao_trabalho = SIT_TRAB,
    encaminhado_caps = CAPES,
    cat_emitida = CAT,
    ocupacao = OCUPACAO,
    regime_tratamento = REGIME,
    periodo_notificacao = PANDEMIA,
    diagnostico = DIAG2,
    outros_casos_trabalho = TRAB_DOE
  ) %>%
  tidyr::drop_na()

# Exportar intermediário -----------------------------------------------------
saveRDS(dados_pp, "dados_preprocessados.rds")

# Split treino/teste (80/20) ------------------------------------------------
split_obj <- rsample::initial_split(dados_pp, prop = 0.8, strata = NULL)

treino <- rsample::training(split_obj) %>% dplyr::mutate(grupo = factor("Treino"))
teste  <- rsample::testing(split_obj)  %>% dplyr::mutate(grupo = factor("Teste"))

dados_pp_grupo <- dplyr::bind_rows(treino, teste)

# Salvar saídas do split
saveRDS(dados_pp_grupo, "dados_preprocessados_com_grupo.rds")
saveRDS(dplyr::select(treino, -grupo), "treino.rds")
saveRDS(dplyr::select(teste,  -grupo), "teste.rds")

# Lê o dataset com 'grupo' (treino/teste)
dados <- readRDS("dados_preprocessados_com_grupo.rds")

# Tabela descritiva por grupo -----------------------------------------------
# Define rótulos legíveis (sem alterar nomes internos)
labels_list <- list(
  faixa_etaria ~ "Faixa etária",
  sexo ~ "Sexo",
  escolaridade ~ "Escolaridade",
  raca_cor ~ "Raça/cor",
  terceirizado ~ "Terceirizado",
  regiao ~ "Região",
  alcool ~ "Consumo habitual de álcool",
  psicofarmacos ~ "Uso de psicofármacos",
  drogas ~ "Uso de drogas",
  fumante ~ "Fumante",
  afastado_trabalho ~ "Foi afastado do trabalho",
  situacao_trabalho ~ "Situação de trabalho",
  encaminhado_caps ~ "Encaminhado ao CAPS",
  cat_emitida ~ "Emitida Comunicação de Acidente de Trabalho (CAT)",
  ocupacao ~ "Ocupação",
  regime_tratamento ~ "Regime de tratamento",
  periodo_notificacao ~ "Período de notificação",
  diagnostico ~ "Diagnóstico",
  outros_casos_trabalho ~ "Outros casos no trabalho"
)

tabela_perfil <- dados %>%
  gtsummary::tbl_summary(
    by = grupo,
    missing = "no",
    label = labels_list
  ) %>%
  gtsummary::add_overall() %>%
  gtsummary::add_p(pvalue_fun = function(x) gtsummary::style_pvalue(x, digits = 2))


# Exporta para .docx
tabela_perfil %>% as_flex_table() %>% save_as_docx(path = "tabela_perfil_treino_teste.docx")

# Testes (exemplos com Fisher, se tabelas são esparsas) ---------------------
# Construir tabelas de contingência de forma robusta:
tab_diag  <- table(dados$grupo, dados$diagnostico, useNA = "no")
tab_ocup  <- table(dados$grupo, dados$ocupacao,    useNA = "no")

# Fisher (usar simulate.p.value=TRUE em tabelas maiores)
teste_fisher_diag <- fisher.test(tab_diag, simulate.p.value = TRUE, B = 1e5)
teste_fisher_ocup <- fisher.test(tab_ocup, simulate.p.value = TRUE, B = 1e5)

