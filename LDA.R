# LDA Assumptions
# Binary outcome (already satisfied)
# Normality within each class
# Equal covariance matrices across classes (homogeneity)
# No multicollinearity
# No missing data
# Sufficient sample size



# ====================== LODA DATA ======================
df <- read.csv2("processed.csv")
df <- df[, -1]  # drops the first column (index)

# check if data load was success
dim(df)

# ====================== SPLIT ======================
set.seed(67) # replicability
index <- createDataPartition(df$target, p = 0.8, list = FALSE)
train_data <- df[index, ]
test_data  <- df[-index, ]


# ====================== SPLIT CONTINUOUS AND DISCRETE ======================
# First separate continues and discrete data
# Lets look at the qqplots of features and decide
# which are continues and discrete
# LDA only work with continous data 


# For Naive Bayes we tryed to select continuous data by hand
# Here we will try algorithmic approach
# for every average unique count >= 5 we define the features 
# as discrete

predictors <- setdiff(names(train_data), "target")

continuous <- c()
discrete <- c()

for (feat in predictors) {
  n_unique <- length(unique(train_data[[feat]]))
  avg_count <- nrow(train_data) / n_unique
  
  if (avg_count >= 5) {
    discrete <- c(discrete, feat)
  } else {
    continuous <- c(continuous, feat)
  }
}

cat("Continuous features:", length(continuous), "\n")
cat("Discrete features:", length(discrete), "\n")
print(discrete)
print(continuous)


train_lda <- train_data[, c("target", continuous)]
test_lda <- test_data[, c("target", continuous)]



# ================ NORMALITY CHECK ================
# We will use both visual techniques and shapiros willk test based on paper
# https://pmc.ncbi.nlm.nih.gov/articles/PMC3693611/

check_normality <- function(df, features){
  results <- data.frame(
    feature = character(),
    shapiro_p = numeric(),
    skewness = numeric(),
    kurtosis = numeric(),
    qq_verdict = character(),
    final_verdict = character(),
    stringsAsFactors = FALSE
  )
  
  pass <- df[, features]
  fail <- df[, features]
  
  for (feat in features) {
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
  return(list(results = results, fail = fail, pass = pass))
}

# split train data into each class
class_0 <- train_lda[train_lda$target == 0, ]
class_1 <- train_lda[train_lda$target == 1, ]


output <- check_normality(class_0, continuous)
fail_class_0 <- output$fail
pass_class_0 <- output$pass

output <- check_normality(class_1, continuous)
fail_class_1 <- output$fail
pass_class_1 <- output$pass


setdiff(names(pass_class_0), names(pass_class_1))
# features that passed normality check in class 1 also passed in class 0
passed <- rbind(pass_class_0, pass_class_1)
failed <- rbind(fail_class_0, fail_class_1)


train_clean <- train_lda[, c("target", names(passed))]
test_clean <- test_lda[, c("target", names(passed))]



# We already dropped highly (>=0.8) correlated predictors in EDA+preprocessing

# ================ VIF CHECK ================
# Iterative VIF removal
threshold <- 5
current_predictors <- setdiff(names(train_clean), "target")
library(car) # VIF
repeat {
  current_formula <- as.formula(
    paste("target ~", paste(current_predictors, collapse = " + "))
  )
  current_model <- glm(current_formula, data = train_clean, family = "binomial")
  
  vif_vals <- vif(current_model)
  max_vif <- max(vif_vals)
  
  if (max_vif <= threshold) break
  
  drop <- names(which.max(vif_vals))
  current_predictors <- setdiff(current_predictors, drop)
}

# Before VIF
length(names(train_lda))
#After VIF cleanup
#Remaining features
length(current_predictors)

train_clean <- train_lda[, c("target", current_predictors)]
test_clean <- test_lda[, c("target", current_predictors)]



# ====================== Homoscedasticity check ======================
predictors <- setdiff(names(train_clean), "target")

class_0 <- train_clean[train_clean$target == 0, ]
class_1 <- train_clean[train_clean$target == 1, ]

# Compare variance of each feature across classes
var_comparison <- data.frame(
  feature = predictors,
  var_class0 = sapply(predictors, function(f) var(class_0[[f]], na.rm = TRUE)),
  var_class1 = sapply(predictors, function(f) var(class_1[[f]], na.rm = TRUE))
)

var_comparison$ratio <- var_comparison$var_class1 / var_comparison$var_class0
# Ratio close to 1 = equal variance
# Ratio > 2 or < 0.5 = problematic

var_comparison <- var_comparison[order(abs(var_comparison$ratio - 1), decreasing = TRUE), ]
print(var_comparison, row.names = FALSE)

cat("\nProblematic (ratio > 2 or < 0.5):",
    sum(var_comparison$ratio > 2 | var_comparison$ratio < 0.5), "\n")

# Visual check: side by side boxplots
pdf("homoscedasticity_check.pdf", width = 12, height = 8)
par(mfrow = c(3, 3))

for (feat in predictors) {
  boxplot(
    train_clean[[feat]] ~ train_clean$target,
    main = feat, cex.main = 0.7,
    names = c("Loss", "Win"),
    col = c("lightblue", "lightcoral")
  )
}
dev.off()


library(biotools)

predictors <- setdiff(names(train_clean), "target")

boxM_result <- boxM(
  train_clean[, predictors],
  train_clean$target
)
print(boxM_result)

# Box's M test rejected the null hypothesis of equal covariance matrices 
# (p < 0.001), however this test is known to be overly sensitive.
# Visual inspection and variance ratio analysis showed all features had variance
# ratios between 0.82 and 1.24, indicating the assumption is practically met.


# ================== FIT MODEL ==================  
library(MASS)

lda_model <- lda(
  as.factor(target) ~ .,
  data = train_clean
)

# Train accuracy
lda_train_pred <- predict(lda_model, train_clean)$class
train_tab <- table(lda_train_pred, as.factor(train_clean$target))
cat("\nTrain Accuracy:", round(sum(diag(train_tab)) / sum(train_tab), 4), "\n")

# Test accuracy
lda_test_pred <- predict(lda_model, test_clean)$class
test_tab <- table(lda_test_pred, as.factor(test_clean$target))
print(test_tab)
cat("Test Accuracy:", round(sum(diag(test_tab)) / sum(test_tab), 4), "\n")

