# XGBoost (caret::train, method = "xgbTree")

# ---------------------------------------------------------------------------
# Entradas (usar arquivos salvos em data/processed/)
# ---------------------------------------------------------------------------

treino <- readRDS("treino.rds")
teste  <- readRDS("teste.rds")

# ---------------------------------------------------------------------------
# Preparação de variáveis
# ---------------------------------------------------------------------------
resp_name <- "afastado_trabalho"
if (!resp_name %in% names(treino) || !resp_name %in% names(teste)) {
  stop(sprintf("A coluna '%s' não foi encontrada em treino/teste.", resp_name))
}

# Garantir codificação binária coerente (níveis na ordem: negativo, positivo)
levels_bin <- c("Não", "Sim")

treino[[resp_name]] <- factor(treino[[resp_name]], levels = levels_bin)
teste[[resp_name]]  <- factor(teste[[resp_name]],  levels = levels_bin)

# ---------------------------------------------------------------------------
# Paralelização
# ---------------------------------------------------------------------------
if (requireNamespace("doParallel", quietly = TRUE)) {
  library(doParallel)
  cl <- parallel::makeCluster(max(1L, parallel::detectCores() - 1L))
  doParallel::registerDoParallel(cl)
  on.exit({ try(parallel::stopCluster(cl), silent = TRUE); doParallel::registerDoSEQ() }, add = TRUE)
}

# ---------------------------------------------------------------------------
# Treinamento com caret::train (xgbTree)
# ---------------------------------------------------------------------------
library(caret)
library(pROC)
library(boot)

ctrl <- caret::trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  allowParallel = TRUE,
  search = "random"
)

# Para buscas longas, ajuste tuneLength conforme recursos. 100–200 costuma ser razoável.
set.seed(123)
modelo_xgb <- caret::train(
  afastado_trabalho ~ .,
  data = treino,
  method = "xgbTree",
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 1000
)

# Persistir modelo e resumo
saveRDS(modelo_xgb, "modelo_xgb_caret.rds")

sink("xgb_resumo.txt")
cat("Resumo do treino (caret::train - xgbTree)\n\n")
print(modelo_xgb)
sink()

# Perfil de tuning
# png("xgb_tuning.png", width = 1400, height = 900, res = 150)
# plot(modelo_xgb)
# dev.off()

# ---------------------------------------------------------------------------
# Avaliação no conjunto de teste
# ---------------------------------------------------------------------------
pred_xgb_class <- predict(modelo_xgb, newdata = teste)
pred_xgb_prob  <- predict(modelo_xgb, newdata = teste, type = "prob")[, "Sim"]

cm_xgb <- caret::confusionMatrix(pred_xgb_class, teste[[resp_name]], positive = "Sim")

sink("xgb_confusion_matrix.txt")
cat("Matriz de confusão (Teste)\n\n")
print(cm_xgb)
sink()

roc_xgb <- pROC::roc(response = teste[[resp_name]],
                     predictor = pred_xgb_prob,
                     levels = levels_bin)

sink("xgb_auc_roc.txt")
cat("AUC-ROC\n\n")
print(roc_xgb)
cat("\nIC95% da AUC (DeLong):\n")
print(pROC::ci.auc(roc_xgb))
sink()

png("xgb_roc_curve.png", width = 1400, height = 900, res = 150)
pROC::plot.roc(roc_xgb, print.thres = TRUE)
dev.off()

sink("xgb_roc_threshold_ci.txt")
cat("IC para sensibilidade e especificidade no threshold = 0.5\n\n")
print(pROC::ci.thresholds(roc_xgb, threshold = 0.5))
sink()

# Métricas adicionais
f1_xgb <- caret::F_meas(pred_xgb_class, teste[[resp_name]], positive = "Sim")
precision_xgb <- cm_xgb$byClass["Precision"]

write.table(
  data.frame(metric = c("Precision", "F1"), value = c(precision_xgb, f1_xgb)),
  file = "xgb_precision_f1.txt",
  row.names = FALSE, quote = FALSE, sep = "\t"
)

# Importância de variáveis (do caret)
imp_xgb <- caret::varImp(modelo_xgb)
try({
  png("xgb_var_importance.png", width = 1400, height = 900, res = 150)
  plot(imp_xgb, main = "Importância das variáveis (XGBoost)")
  dev.off()
}, silent = TRUE)

# ---------------------------------------------------------------------------
# IC95% para Precisão e F1 (bootstrap estratificado)
# ---------------------------------------------------------------------------
set.seed(123)
n_boot <- 2000
true_labels <- teste[[resp_name]]

calc_metrics_fixed <- function(data, indices) {
  true_b <- true_labels[indices]
  pred_b <- pred_xgb_class[indices]
  cm_b <- caret::confusionMatrix(pred_b, true_b, positive = "Sim")
  precision_b <- cm_b$byClass["Precision"]
  f1_b <- caret::F_meas(pred_b, true_b, positive = "Sim")
  c(precision = precision_b, f1 = f1_b)
}

boot_results <- boot::boot(
  data = teste,
  statistic = calc_metrics_fixed,
  R = n_boot,
  strata = true_labels
)

ci_precision <- boot::boot.ci(boot_results, type = "perc", index = 1)
ci_f1        <- boot::boot.ci(boot_results, type = "perc", index = 2)

ci_table <- data.frame(
  metric = c("Precision", "F1"),
  low    = c(ci_precision$percent[4], ci_f1$percent[4]),
  median = apply(boot_results$t, 2, stats::median, na.rm = TRUE),
  high   = c(ci_precision$percent[5], ci_f1$percent[5])
)

write.table(ci_table,
            file = "xgb_ci_bootstrap_precision_f1.txt",
            row.names = FALSE, quote = FALSE, sep = "\t")

# ---------------------------------------------------------------------------
# SHAP com fastshap + shapviz (explicações no conjunto de teste)
# ---------------------------------------------------------------------------
if (requireNamespace("fastshap", quietly = TRUE) && requireNamespace("shapviz", quietly = TRUE)) {
  library(dplyr)
  library(fastshap)
  library(shapviz)
  message("Gerando explicações SHAP (fastshap/shapviz)...")
  
  x_train <- treino %>% dplyr::select(-dplyr::all_of(resp_name)) %>% as.data.frame()
  x_test  <- teste  %>% dplyr::select(-dplyr::all_of(resp_name)) %>% as.data.frame()
  y_test  <- as.numeric(teste[[resp_name]] == "Sim")  # opcional, para shapviz
  
  # Wrapper: retorna probabilidade da classe positiva "Sim"
  pred_wrapper <- function(object, newdata) {
    predict(modelo_xgb, newdata = newdata, type = "prob")[, "Sim"]
  }
  
  set.seed(123)
  shap_values <- fastshap::explain(
    object       = modelo_xgb,
    X            = x_train,
    pred_wrapper = pred_wrapper,
    newdata      = x_test,
    nsim         = 300
  )
  
  baseline <- mean(pred_wrapper(modelo_xgb, x_train))
  sv <- shapviz::shapviz(shap_values, X = x_test, y = y_test, baseline = baseline)
  
  var_map_shap <- c(
    "faixa_etaria"          = "Age group",
    "sexo"                  = "Biological sex",
    "raca_cor"              = "Race/ethnicity",
    "escolaridade"          = "Education",
    "situacao_trabalho"     = "Employment status",
    "terceirizado"          = "Outsourced employment",
    "regime_tratamento"     = "Treatment regime",
    "alcool"                = "Alcohol use",
    "psicofarmacos"         = "Psychotropic medication",
    "drogas"                = "Illicit drug use",
    "fumante"               = "Smoking status",
    "outros_casos_trabalho" = "Other workers affected",
    "encaminhado_caps"      = "Referred to CAPS",
    "cat_emitida"           = "Work accident report",
    "ocupacao"              = "Occupation",
    "regiao"                = "Geographic region",
    "periodo_notificacao"   = "Notification period",
    "diagnostico"           = "Diagnosis (ICD-10)"
  )
  
  # função para renomear dentro do objeto shapviz
  rename_shapviz <- function(sv, dict) {
    sv2 <- sv
    old <- colnames(sv2$S)
    new <- vapply(old, function(x) if (x %in% names(dict)) dict[[x]] else x, character(1))
    colnames(sv2$S) <- new
    if (!is.null(sv2$S_abs)) colnames(sv2$S_abs) <- new
    if (!is.null(sv2$X))     colnames(sv2$X)     <- new
    # alguns objetos têm este campo; se existir, atualize
    if (!is.null(sv2$feature_names)) sv2$feature_names <- new
    sv2
  }
  
  sv_en <- rename_shapviz(sv, var_map_shap)
  
  # Mapeamentos de níveis por coluna (use apenas os que se aplicam ao seu dataset)
  lvl_map <- list(
    "Age group" = c(
      "18-29" = "18–29",
      "30-39" = "30–39",
      "40-49" = "40–49",
      "50-59" = "50–59",
      "60 ou mais" = "≥60"
    ),
    "Biological sex" = c(
      "F" = "Female",
      "M" = "Male"
    ),
    "Race/ethnicity" = c(
      "Branca" = "White",
      "Preta/Parda" = "Black/Brown",
      "Amarela/Indígena" = "Asian/Indigenous"
    ),
    "Education" = c(
      "Até ensino fundamental" = "Up to primary",
      "Ensino fundamental"     = "Primary",
      "Ensino médio"           = "Secondary",
      "Ensino superior"        = "Higher"
    ),
    "Employment status" = c(
      "Empregado com carteira"  = "Formal employee",
      "Empregado sem carteira"  = "Informal employee",
      "Servidor público"        = "Public servant",
      "Autônomo"                = "Self-employed",
      "Aposentado"              = "Retired",
      "Desempregado"            = "Unemployed",
      "Outros"                  = "Other"
    ),
    "Outsourced employment" = c("Sim"="Yes","Não"="No"),
    "Treatment regime"      = c("Ambulatorial"="Outpatient","Hospitalar"="Inpatient"),
    "Alcohol use"           = c("Sim"="Yes","Não"="No"),
    "Psychotropic medication" = c("Sim"="Yes","Não"="No"),
    "Illicit drug use"      = c("Sim"="Yes","Não"="No"),
    "Smoking status"        = c("Sim"="Yes","Não"="No"),
    "Other workers affected"= c("Sim"="Yes","Não"="No"),
    "Referred to CAPS"      = c("Sim"="Yes","Não"="No"),
    "Work accident report"  = c("Sim"="Yes","Não"="No"),
    "Occupation" = c(
      "Ciências/artes"                                           = "Science/arts",
      "Comércio"                                                 = "Commerce",
      "Bens e serviços industriais"                              = "Industrial goods/services",
      "Técnicos de nível médio"                                  = "Mid-level technicians",
      "Forças armadas/Policiais/Bombeiros militares"             = "Military/police/firefighters",
      "Membros superiores do poder público/interesse público"    = "Public administration",
      "Reparação e manutenção"                                   = "Repair/maintenance",
      "Serviços administrativos"                                 = "Administrative services",
      "Agropecuários/ florestais/ pesca"                         = "Agriculture/forestry/fishing"
    ),
    "Geographic region" = c(
      "Norte"        = "North",
      "Nordeste"     = "Northeast",
      "Centro-Oeste" = "Central-West",
      "Sudeste"      = "Southeast",
      "Sul"          = "South"
    ),
    "Notification period" = c(
      "Pré-pandemia" = "Pre-pandemic",
      "Pandemia"     = "Pandemic"
    ),
    "Diagnosis (ICD-10)" = c(
      "Transtornos do humor (afetivos)"                                   = "Mood disorders",
      "Transtornos neuróticos, relacionados ao estresse e somatoformes"   = "Neurotic/stress-related disorders",
      "Transtornos esquizofrênicos, esquizotípicos e delirantes"          = "Schizophrenic spectrum disorders",
      "Transtornos de personalidade e comportamentais"                     = "Personality disorders",
      "Transtorno mental não especificado"                                 = "Unspecified mental disorder",
      "Síndrome de Burnout"                                                = "Burnout syndrome"
    )
  )
  
  recode_factor_levels <- function(df, maps) {
    for (col in names(maps)) {
      if (col %in% names(df) && is.factor(df[[col]])) {
        # para cada par PT->EN no mapa daquela coluna
        m <- maps[[col]]
        # troca de níveis de forma segura (somente se o nível existe)
        lv <- levels(df[[col]])
        for (old in names(m)) {
          lv[lv == old] <- m[[old]]
        }
        levels(df[[col]]) <- lv
        # opcional: dropar níveis não usados (se precisar)
        df[[col]] <- droplevels(df[[col]])
      }
    }
    df
  }
  
  sv_en$X <- recode_factor_levels(sv_en$X, lvl_map)

  # Ex.: waterfall para uma observação
  sv_waterfall(sv_en, row = 1)
  
  # agora os gráficos saem em inglês
  shapviz::sv_importance(sv_en, kind = "beeswarm") # beeswarm
  sv_importance(sv_en)        # barras de importância (mean |SHAP|)
  sv_waterfall(sv_en, row=1)  # exemplo de indivíduo
  
  # Importância global
  try({
    png("xgb_shap_importance.png", width = 20, height = 20, res = 600,
        units = "cm")
    print(shapviz::sv_importance(sv_en))
    dev.off()
  }, silent = TRUE)
  
  # Beeswarm
  try({
    png("xgb_shap_beeswarm.png", width = 20, height = 20, res = 600,
        units = "cm")
    print(shapviz::sv_importance(sv_en, kind = "beeswarm"))
    dev.off()
  }, silent = TRUE)
  
}

library(shapviz)
library(ggplot2)
library(patchwork)

# escolha os IDs desejados
ids <- c(531, 270, 396, 376)

# gerar 4 gráficos
p1 <- sv_waterfall(sv_en, row_id = ids[1])
p2 <- sv_waterfall(sv_en, row_id = ids[2])
p3 <- sv_waterfall(sv_en, row_id = ids[3])
p4 <- sv_waterfall(sv_en, row_id = ids[4])

# combinar 2x2
combined <- (p1 | p2) /
  (p3 | p4)

# salvar
png("xgb_shap_waterfalls_4panel.png", width = 40, height = 25, 
    res = 600, units = "cm")
print(combined)
dev.off()
