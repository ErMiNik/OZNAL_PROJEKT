# Normality check

library(e1071)

predictors <- setdiff(names(train_data), "team100_win")

results <- data.frame(
  feature = character(),
  shapiro_p = numeric(),
  skewness = numeric(),
  kurtosis = numeric(),
  qq_verdict = character(),
  final_verdict = character(),
  stringsAsFactors = FALSE
)


pass <- train_data
fail <- train_data

for (feat in predictors) {
  x <- train_data[[feat]][!is.na(train_data[[feat]])]
  
  sh <- shapiro.test(x)
  sk <- skewness(x)
  kt <- kurtosis(x)
  
  # Practical normality from skewness/kurtosis
  practical <- abs(sk) < 1 & abs(kt) < 3
  if (practical == F)
  {pass[feat] = NULL
  }else{
    fail[feat] = NULL
  }
  
  # Combined verdict:
  # If Shapiro PASSES -> normal
  # If Shapiro FAILS but practical thresholds pass -> practically normal
  # If both fail -> not normal
  verdict <- ifelse(
    sh$p.value > 0.05, "Normal",
    ifelse(practical, "Practically normal", "Not normal")
  )
  
  results <- rbind(results, data.frame(
    feature = feat,
    shapiro_p = round(sh$p.value, 4),
    skewness = round(sk, 3),
    kurtosis = round(kt, 3),
    practical = practical,
    final_verdict = verdict,
    stringsAsFactors = FALSE
  ))
}

dim(train_data)
dim(pass)
dim(fail)





# 3. Q-Q plots for visual inspection (saves to PDF)
pdf("qq_fail.pdf", width = 12, height = 10)
par(mfrow = c(3, 3))
predictors <- names(fail)
for (feat in predictors) {
  qqnorm(fail[[feat]], main = feat, cex.main = 0.8)
  qqline(fail[[feat]], col = "red")
}
dev.off()

fet <- train_data$team100_double_kills
min_max_norm <- (fet - min(fet)/(max(fet)-min(fet)))
mean_norm <- (fet - mean(fet)) / (max(fet)-min(fet))


hist(mean_norm, prob=TRUE)

hist(fet)
hist(log1p(fet))
curve(dnorm(fet, mean=mean(fet)), add=T)

boxplot(fet)
boxplot(mean_norm)

ddd <- log1p(train_data$team200_total_enemy_jungle_minions_killed)
hh <- train_data$team200_total_enemy_jungle_minions_killed

boxplot(hh)

qqnorm(ddd)
qqline(ddd, col="red")
shapiro.test(mean_norm)
hist((fet - min(fet)/(max(fet)-min(fet))))
skewness(train_data$team100_double_kills)
kurtosis(train_data$team100_double_kills)
boxplot(train_data$team100_double_kills)


str(results)
temp <- results$practical == T


cat("\n--- Summary ---\n")
cat("Normal (Shapiro passes):        ", sum(results$final_verdict == "Normal"), "\n")
cat("Practically normal (plots OK):  ", sum(results$final_verdict == "Practically normal"), "\n")
cat("Not normal (both fail):         ", sum(results$final_verdict == "Not normal"), "\n")


# Recalculate with actual training data
n_train <- nrow(train_data)
n_events <- min(table(train_data$team100_win))
n_predictors <- length(setdiff(names(train_data), "team100_win"))
epp <- n_events / n_predictors

cat("Training set size:", n_train, "\n")
cat("Minority class count:", n_events, "\n")
cat("Number of predictors:", n_predictors, "\n")
cat("EPP:", round(epp, 1), "\n")
cat("Target: EPP >= 10\n")
cat("Max predictors you can afford:", floor(n_events / 10), "\n")



# Assuming your training data is called 'train_data'
# Separate predictors (exclude target variable)
predictors <- setdiff(names(train_data), "team100_win")

# 1. Shapiro-Wilk test for each feature
# (uses sample of 5000 if n > 5000, since Shapiro-Wilk has a limit)
shapiro_results <- data.frame(
  feature = character(),
  statistic = numeric(),
  p_value = numeric(),
  normal = character(),
  stringsAsFactors = FALSE
)

for (feat in predictors) {
  x <- train_data[[feat]]
  x <- x[!is.na(x)]
  
  if (length(x) > 5000) {
    set.seed(42)
    x <- sample(x, 5000)
  }
  
  test <- shapiro.test(x)
  shapiro_results <- rbind(shapiro_results, data.frame(
    feature = feat,
    statistic = round(test$statistic, 4),
    p_value = test$p.value,
    normal = ifelse(test$p.value > 0.05, "Yes", "No"),
    stringsAsFactors = FALSE
  ))
}

shapiro_results <- shapiro_results[order(shapiro_results$p_value, decreasing = TRUE), ]
print(shapiro_results, row.names = FALSE)

cat("\n--- Summary ---\n")
cat("Normal (p > 0.05):", sum(shapiro_results$normal == "Yes"), "\n")
cat("Not normal (p <= 0.05):", sum(shapiro_results$normal == "No"), "\n")

# 2. Skewness and kurtosis
library(e1071)

sk_results <- data.frame(
  feature = predictors,
  skewness = sapply(predictors, function(f) round(skewness(train_data[[f]], na.rm = TRUE), 3)),
  kurtosis = sapply(predictors, function(f) round(kurtosis(train_data[[f]], na.rm = TRUE), 3)),
  row.names = NULL
)

cat("\n--- Skewness & Kurtosis ---\n")
print(sk_results, row.names = FALSE)

# 3. Q-Q plots for visual inspection (saves to PDF)
pdf("qq_plots.pdf", width = 12, height = 10)
par(mfrow = c(3, 3))
for (feat in predictors) {
  qqnorm(train_data[[feat]], main = feat, cex.main = 0.8)
  qqline(train_data[[feat]], col = "red")
}
dev.off()
cat("\nQ-Q plots saved to qq_plots.pdf\n")


dim(train_data)

shapiro.test(train_data$team100_vision_score_per_minute)

shapiro.test(train_data$team100_kills)