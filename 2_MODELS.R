library(caret)
library(car)
library(MASS)
library(e1071)

# Load data
df <- read.csv2("processed.csv")
df <- df[, -1]  # drops the first column (index)

# Make target a factor
df$team100_win <- as.factor(df$team100_win)

# --- SPLIT ---
set.seed(123)
index <- createDataPartition(df$team100_win, p = 0.8, list = FALSE)
train <- df[index, ]
test  <- df[-index, ]

# =====================
# CHECKS ON TRAIN ONLY
# =====================

# 1. Class balance
cat("Class balance:\n")
print(table(train$team100_win))
print(prop.table(table(train$team100_win)))

# 2. Check for near-zero variance features
nzv <- nearZeroVar(train, saveMetrics = TRUE)
cat("\nNear-zero variance features:\n")
print(nzv[nzv$nzv == TRUE, ])

# 3. Check normality (for LDA) — test a few features
cat("\nShapiro-Wilk tests (per class, first 5 numeric features):\n")
numeric_cols <- names(train)[sapply(train, is.numeric)]
for (col in numeric_cols[1:5]) {
  for (cls in levels(train$team100_win)) {
    subset_data <- train[[col]][train$team100_win == cls]
    # Shapiro only works for n <= 5000, sample if needed
    if (length(subset_data) > 5000) subset_data <- sample(subset_data, 5000)
    p <- shapiro.test(subset_data)$p.value
    cat(sprintf("  %s | class %s | p = %.4f %s\n", col, cls, p, ifelse(p < 0.05, "* NOT NORMAL", "")))
  }
}

# 4. Box's M test for equal covariance (for LDA)
# Can be slow with many features, use a subset
cat("\nBox's M test (first 10 features):\n")
tryCatch({
  library(biotools)
  boxM(train[, numeric_cols[1:10]], train$team100_win)
}, error = function(e) cat("  Could not run Box's M test:", e$message, "\n"))

# 5. VIF check (for Logistic Regression)
cat("\nVIF check:\n")
tryCatch({
  log_model_vif <- glm(team100_win ~ ., data = train, family = binomial)
  vif_values <- vif(log_model_vif)
  high_vif <- vif_values[vif_values > 5]
  if (length(high_vif) > 0) {
    cat("  Features with VIF > 5:\n")
    print(sort(high_vif, decreasing = TRUE))
  } else {
    cat("  All VIF values < 5. No multicollinearity issues.\n")
  }
}, error = function(e) cat("  VIF error:", e$message, "\n"))

# 6. Check for perfect separation (logistic regression)
cat("\nChecking for zero-variance features within classes:\n")
for (col in numeric_cols) {
  for (cls in levels(train$team100_win)) {
    if (sd(train[[col]][train$team100_win == cls]) == 0) {
      cat(sprintf("  WARNING: %s has zero variance in class %s\n", col, cls))
    }
  }
}
