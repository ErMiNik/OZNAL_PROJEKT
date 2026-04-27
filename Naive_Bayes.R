# Assumption of Naive Bayes
# source: https://www.geeksforgeeks.org/machine-learning/naive-bayes-classifiers/


# Feature independence
# - this is a "Naive" assumption that is violated in real world
# - we already dropped highly correlated features
# - so we will just move on

# Continuous features are normally distributed
# - we have to check normality for each class

# Discrete features have multinomial distributions
# - https://faculty.washington.edu/yenchic/20A_stat512/Lec7_Multinomial.pdf
# Features are equally important
# No missing data


library(e1071) #naiveBayes
library(caret) #createDataPartition

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

par(mfrow = c(3, 3))
predictors <- names(train_data)
for (feat in predictors) {
  qqnorm(train_data[[feat]], main = feat, cex.main = 0.8)
  qqline(train_data[[feat]], col = "red")
}

discrete <- c("target", "diff_triple_kills", "diff_largest_multi_kill",
              "diff_dragon_kills", "diff_baron_kills",
              "diff_objectives_stolen", "diff_first_blood_kill",
              "diff_first_blood_assist", "diff_first_tower_kill",
              "diff_first_tower_assist", "diff_wards_guarded")

continuous <- setdiff(names(train_data), discrete)

for (feat in discrete) {
  cat(feat, length(unique(train_data[[feat]])), "\n")
}

for (feat in continuous) {
  cat(feat, length(unique(train_data[[feat]])), "\n")
}


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
class_0 <- train_data[train_data$target == 0, ]
class_1 <- train_data[train_data$target == 1, ]


output <- check_normality(class_0, continuous)
fail_class_0 <- output$fail
pass_class_0 <- output$pass

output <- check_normality(class_1, continuous)
fail_class_1 <- output$fail
pass_class_1 <- output$pass


qqplot_pdf <- function(df){
  
  pdf(sprintf("qq_%s.pdf",deparse(substitute(df))), width = 12, height = 10)
  par(mfrow = c(3, 3))
  predictors <- names(df)
  for (feat in predictors) {
    qqnorm(df[[feat]], main = feat, cex.main = 0.8)
    qqline(df[[feat]], col = "red")
  }
  dev.off()
}

setdiff(names(pass_class_0), names(pass_class_1))
setdiff(names(fail_class_0), names(fail_class_1))
# features that passed normality check in class 1 also passed in class 0

passed <- rbind(pass_class_0, pass_class_1)
failed <- rbind(fail_class_0, fail_class_1)
# we can check all of them together
qqplot_pdf(passed)
qqplot_pdf(failed)

union(names(passed), discrete)

train_clean <- train_data[, union(names(passed), discrete)]
test_clean <- test_data[, union(names(passed), discrete)]


# fit the model
nb_model <- naiveBayes(
  as.factor(target) ~ .,
  data = train_clean,
  laplace = 1 # prevents a single rare value from dominating prediction
)

# Predict on test set
nb_pred <- predict(nb_model, newdata = test_clean, type = "class")
nb_prob <- predict(nb_model, newdata = test_clean, type = "raw")

# Evaluate
tab <- table(nb_pred, as.factor(test_clean$target))
print(tab)

accuracy <- sum(diag(tab)) / sum(tab)
cat("Accuracy:", round(accuracy, 4), "\n")


# Predict on training data
nb_train_pred <- predict(nb_model, newdata = train_clean, type = "class")

# Accuracy
tab_train <- table(nb_train_pred, as.factor(train_clean$target))
print(tab_train)

train_accuracy <- sum(diag(tab_train)) / sum(tab_train)
cat("Train Accuracy:", round(train_accuracy, 4), "\n")

