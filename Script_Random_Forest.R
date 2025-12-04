# Random Forest (caret + randomForest) com validação cruzada, AUC-ROC,
---------------------------------------------------------------------------
# Entradas 
# ---------------------------------------------------------------------------
treino2 <- readRDS("treino_2.rds")
teste2  <- readRDS("teste_2.rds")

# ---------------------------------------------------------------------------
# Preparação de variáveis
# ---------------------------------------------------------------------------
if (!"afast_trab" %in% names(treino2) || !"afast_trab" %in% names(teste2)) {
  stop("A coluna 'afast_trab' não foi encontrada em treino2/teste2.")
}

# Níveis binários coerentes (positivo = "Sim")
levels_bin <- c("Não", "Sim")

treino2$afast_trab <- factor(treino2$afast_trab, levels = levels_bin)
teste2$afast_trab  <- factor(teste2$afast_trab,  levels = levels_bin)

# Opcional: garantir fatores não ordenados para variáveis categóricas específicas
fact_cols <- c("faixa_etaria", "escolaridade")
for (vv in intersect(fact_cols, names(treino2))) treino2[[vv]] <- factor(treino2[[vv]], ordered = FALSE)
for (vv in intersect(fact_cols, names(teste2)))  teste2[[vv]]  <- factor(teste2[[vv]],  ordered = FALSE)

# ---------------------------------------------------------------------------
# Treinamento com caret::train (randomForest)
# ---------------------------------------------------------------------------
library(caret)
library(randomForest)
library(pROC)
library(boot)

ctrl <- caret::trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

set.seed(123)
modelo_rf <- caret::train(
  afast_trab ~ .,
  data = treino2,
  method = "rf",
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 1000,
  importance = TRUE
)

saveRDS(modelo_rf, "modelo_random_forest.rds")

sink("modelo_rf_resumo.txt")
cat("Resumo do treino (caret::train)\n\n")
print(modelo_rf)
cat("\nMelhor mtry:", modelo_rf$bestTune$mtry, "\n")
sink()

png("rf_tuning_mtry.png", width = 1400, height = 900, res = 150)
plot(modelo_rf)
dev.off()

# ---------------------------------------------------------------------------
# Avaliação em teste
# ---------------------------------------------------------------------------
pred_rf_class <- predict(modelo_rf, newdata = teste2)
pred_rf_prob  <- predict(modelo_rf, newdata = teste2, type = "prob")[, "Sim"]

cm_rf <- caret::confusionMatrix(pred_rf_class, teste2$afast_trab, positive = "Sim")

sink("rf_confusion_matrix.txt")
cat("Matriz de confusão (Teste)\n\n")
print(cm_rf)
sink()

roc_rf <- pROC::roc(response = teste2$afast_trab,
                    predictor = pred_rf_prob)

sink("rf_auc_roc.txt")
cat("AUC-ROC\n\n")
print(roc_rf)
cat("\nIC95% da AUC (DeLong):\n")
print(pROC::ci.auc(roc_rf))
sink()

png("rf_roc_curve.png", width = 1400, height = 900, res = 150)
pROC::plot.roc(roc_rf, print.thres = TRUE)
dev.off()

sink("rf_roc_threshold_ci.txt")
cat("IC para sensibilidade e especificidade no threshold = 0.5\n\n")
print(pROC::ci.thresholds(roc_rf, threshold = 0.5))
sink()

# Métricas adicionais
f1_rf <- caret::F_meas(pred_rf_class, teste2$afast_trab, positive = "Sim")
precision_rf <- cm_rf$byClass["Precision"]

write.table(
  data.frame(metric = c("Precision", "F1"), value = c(precision_rf, f1_rf)),
  file = "rf_precision_f1.txt",
  row.names = FALSE, quote = FALSE, sep = "\t"
)

# Importância de variáveis
imp_rf <- caret::varImp(modelo_rf)
try({
  png("rf_var_importance.png", width = 1400, height = 900, res = 150)
  plot(imp_rf, main = "Importância das variáveis (Random Forest)")
  dev.off()
}, silent = TRUE)

# ---------------------------------------------------------------------------
# IC95% para Precisão e F1 (bootstrap estratificado)
# ---------------------------------------------------------------------------
set.seed(123)
n_boot <- 2000
true_labels <- teste2$afast_trab

calc_metrics_rf <- function(data, indices) {
  true_b <- true_labels[indices]
  pred_b <- pred_rf_class[indices]
  cm_b <- caret::confusionMatrix(pred_b, true_b, positive = "Sim")
  precision_b <- cm_b$byClass["Precision"]
  f1_b <- caret::F_meas(pred_b, true_b, positive = "Sim")
  c(precision = precision_b, f1 = f1_b)
}

boot_results_rf <- boot::boot(
  data = teste2,
  statistic = calc_metrics_rf,
  R = n_boot,
  strata = true_labels
)

ci_precision_rf <- boot::boot.ci(boot_results_rf, type = "perc", index = 1)
ci_f1_rf        <- boot::boot.ci(boot_results_rf, type = "perc", index = 2)

ci_table_rf <- data.frame(
  metric = c("Precision", "F1"),
  low    = c(ci_precision_rf$percent[4], ci_f1_rf$percent[4]),
  median = apply(boot_results_rf$t, 2, stats::median, na.rm = TRUE),
  high   = c(ci_precision_rf$percent[5], ci_f1_rf$percent[5])
)

write.table(ci_table_rf,
            file = "rf_ci_bootstrap_precision_f1.txt",
            row.names = FALSE, quote = FALSE, sep = "\t")

