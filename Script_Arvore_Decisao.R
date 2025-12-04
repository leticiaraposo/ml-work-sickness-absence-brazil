# Árvore de decisão (caret + rpart) com validação cruzada, métricas, AUC-ROC,
# ---------------------------------------------------------------------------
# Entradas (usar arquivos salvos em data/processed/)
# ---------------------------------------------------------------------------
treino2 <- readRDS("treino_2.rds")
teste2  <- readRDS("teste_2.rds")

# ---------------------------------------------------------------------------
# Preparação de variáveis
# ---------------------------------------------------------------------------
# Garantir que a variável resposta exista
if (!"afast_trab" %in% names(treino2) || !"afast_trab" %in% names(teste2)) {
  stop("A coluna 'afast_trab' não foi encontrada em treino2/teste2.")
}

# Assegurar codificação binária coerente para o caret (nível positivo = "Sim")
# Ajuste os níveis conforme sua codificação real.
levels_bin <- c("Não", "Sim")

treino2$afast_trab <- factor(treino2$afast_trab, levels = levels_bin)
teste2$afast_trab  <- factor(teste2$afast_trab,  levels = levels_bin)

# Exemplo de fatores ordenados como não ordenados (se necessário)
fact_cols <- c("faixa_etaria", "escolaridade")
for (vv in intersect(fact_cols, names(treino2))) {
  treino2[[vv]] <- factor(treino2[[vv]], ordered = FALSE)
}
for (vv in intersect(fact_cols, names(teste2))) {
  teste2[[vv]] <- factor(teste2[[vv]], ordered = FALSE)
}

levels(treino2$afast_trab) <- c("No", "Yes")
levels(teste2$afast_trab) <- c("No", "Yes")

# ---------------------------------------------------------------------------
# Treinamento com caret::train (rpart)
# ---------------------------------------------------------------------------
library(caret)
library(rpart)
library(rpart.plot)
library(pROC)
library(boot)

ctrl <- caret::trainControl(
  method = "cv",                # k-fold
  number = 10,                   # 10-fold CV
  classProbs = TRUE,             # necessário para ROC
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

set.seed(123)
modelo_arvore <- caret::train(
  afast_trab ~ .,                # resposta vs. todas as preditoras
  data = treino2,
  method = "rpart",
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 1000
)

# Persistir modelo
saveRDS(modelo_arvore, "modelo_arvore_rpart.rds")

# Relato do melhor modelo
sink("modelo_arvore_resumo.txt")
cat("Resumo do treino (caret::train)\n\n")
print(modelo_arvore)
cat("\nMelhor cp:", modelo_arvore$bestTune$cp, "\n")
sink()

# Visualização do perfil de tuning
png("tuning_cp_caret.png", width = 1400, height = 900, res = 150)
plot(modelo_arvore)
dev.off()

# ---------------------------------------------------------------------------
# Avaliação em teste
# ---------------------------------------------------------------------------
pred_class <- predict(modelo_arvore, newdata = teste2)
pred_prob  <- predict(modelo_arvore, newdata = teste2, type = "prob")[, "Yes"]

cm <- caret::confusionMatrix(pred_class, teste2$afast_trab, positive = "Yes")

sink("confusion_matrix.txt")
cat("Matriz de confusão (Teste)\n\n")
print(cm)
sink()

# AUC-ROC
roc_obj <- pROC::roc(response = teste2$afast_trab,
                     predictor = pred_prob)

sink("auc_roc.txt")
cat("AUC-ROC\n\n")
print(roc_obj)
cat("\nIC95% da AUC (DeLong):\n")
print(pROC::ci.auc(roc_obj))
sink()

png("roc_curve.png", width = 1400, height = 900, res = 150)
pROC::plot.roc(roc_obj, print.thres = TRUE)
dev.off()

# Threshold fixo 0.5 (IC dos thresholds)
sink("roc_threshold_ci.txt")
cat("IC para sensibilidade e especificidade no threshold = 0.5\n\n")
print(pROC::ci.thresholds(roc_obj, threshold = 0.5))
sink()

# Métricas adicionais
f1_value <- caret::F_meas(pred_class, teste2$afast_trab, positive = "Yes")
precisao <- cm$byClass["Precision"]

write.table(
  data.frame(metric = c("Precision", "F1"), value = c(precisao, f1_value)),
  file = "precision_f1.txt",
  row.names = FALSE, quote = FALSE, sep = "\t"
)

# Importância de variáveis (do caret)
imp <- caret::varImp(modelo_arvore)
try({
  png("var_importance.png", width = 1400, height = 900, res = 150)
  plot(imp, main = "Importância das variáveis (rpart)")
  dev.off()
}, silent = TRUE)

# Plot da árvore final
try({
  png("rpart_tree.png", width = 1600, height = 1200, res = 150)
  rpart.plot::rpart.plot(modelo_arvore$finalModel, type = 2, extra = 104, cex = 0.8)
  dev.off()
}, silent = TRUE)

library(rpart)
library(rpart.plot)

tree <- modelo_arvore$finalModel

# --- 2) Mapa PT -> EN para nomes de variáveis dos splits ---
var_map <- c(
  "emitido_catSim" = "Work accident report",
  "regiaoNordeste" = "Northeast",
  "regiaoSul" = "South",
  "encam_capesSim" = "Referred to CAPS",
  "uso_psico_farmSim" = "Psychotropic meds",
  "pandemiaPré-pandemia" = "Pre-pandemic",
  "trab_doeSim" = "Other workers affected",
  "escolaridadeEnsino médio" = "Secondary educ.",
  "escolaridadeEnsino superior" = "Higher educ.",
  "sit_trabOutros" = "Employm.: Other",
  "sit_trabServidor público" = "Public servant",
  "ocupacaoServiços administrativos" = "Admin services",
  "ocupacaoTécnicos de nível médio" = "Mid-level tech.",
  "diagnosticoTranstornos do humor (afetivos)" = "ICD-10: Mood",
  "diagnosticoTranstornos neuróticos, relacionados ao estresse e somatoformes" = "ICD-10: Neurotic/stress",
  "uso_drogasSim" = "Illicit drugs",
  "uso_alcoolSim" = "Alcohol",
  "faixa_etaria40-49" = "Age 40–49",
  "faixa_etaria50-59" = "Age 50–59",
  "terceirizadoSim" = "Outsourced",
  "ocupacaoComércio" = "Commerce"
)


pretty_names <- function(x){
  for(nm in names(var_map)) x <- gsub(nm, var_map[[nm]], x, fixed = TRUE)
  x
}

# --- 3) Plot seguro ---
png("DT_english.png", width = 120, height = 60, units = "cm", res = 600)
prp(
  tree,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  faclen = 3, varlen = 0,
  cex = 1.1,
  split.cex = 1.3,
  under.cex = 1.3,
  box.palette = 0,
  family = "sans",
  split.fun = function(x, labs, digits, varlen, faclen) pretty_names(labs)
)
dev.off()


# ---------------------------------------------------------------------------
# IC95% para Precisão e F1 (bootstrap estratificado no conjunto de teste)
# ---------------------------------------------------------------------------
set.seed(123)
n_boot <- 2000
true_labels <- teste2$afast_trab

# Função de estatística: reamostra índices do conjunto de teste e recalcula métricas
calc_metrics_arvore <- function(data, indices) {
  true_b <- true_labels[indices]
  pred_b <- pred_class[indices]
  cm_b <- caret::confusionMatrix(pred_b, true_b, positive = "Sim")
  precision_b <- cm_b$byClass["Precision"]
  f1_b <- caret::F_meas(pred_b, true_b, positive = "Sim")
  c(precision = precision_b, f1 = f1_b)
}

boot_results <- boot::boot(
  data = teste2,
  statistic = calc_metrics_arvore,
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
            "ci_bootstrap_precision_f1.txt",
            row.names = FALSE, quote = FALSE, sep = "\t")

