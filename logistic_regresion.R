library(caret) #createDataPartition
library(car) # model

# Assumptions of Logicstic Regression

# 1. Independent observations
# 2. Binary dependent variables
# 3. Linearity relationship between independent variables and log odds
# 4. No extreme outliers
# 5. Large sample size
# 6. Colinearity

# 1. each observation is a different Match - they are independent
# 2. Target is binary (team100 1-win 0-lose -> team200 won)
# 3. will check this later
# 4. will deal with this later
# 5. we have ~1000 rows in train dataset
# 6. will deal with this later


# We have to deal with 3. 4. and 6.

# Load data
df <- read.csv2("processed.csv")
df <- df[, -1]  # drops the first column (index)

# check if data load was success
dim(df)


# --- SPLIT ---
set.seed(67) # replicability
index <- createDataPartition(df$team100_win, p = 0.8, list = FALSE)
train_data <- df[index, ]
test_data  <- df[-index, ]


# We dropped highly (>=0.8) correlated predictors in EDA+preprocessing
# After consulting with lecturer we remove very strong predictors
# which are turret and inhibitor data


to_drop <- c("team100_turret_takedowns", "team100_inhibitor_takedowns",
             "team200_turret_takedowns", "team200_inhibitor_kills")

train_data <- train_data[, !names(train_data) %in% to_drop]
test_data <- test_data[, !names(test_data) %in% to_drop]


# Iterative VIF removal
threshold <- 5
current_predictors <- setdiff(names(train_data), "team100_win")

repeat {
  current_formula <- as.formula(
    paste("team100_win ~", paste(current_predictors, collapse = " + "))
  )
  current_model <- glm(current_formula, data = train_data, family = "binomial")
  
  vif_vals <- vif(current_model)
  max_vif <- max(vif_vals)
  
  if (max_vif <= threshold) break
  
  drop <- names(which.max(vif_vals))
  current_predictors <- setdiff(current_predictors, drop)
}

#After VIF cleanup
#Remaining features
length(current_predictors)
# Converged
current_model$converged

train_data <- train_data[, c("team100_win", current_predictors)]
test_data <- test_data[, c("team100_win", current_predictors)]



predictors <- setdiff(names(train_data), "team100_win")

par(mfrow = c(3, 3))

for (feat in predictors) {
  x <- train_data[[feat]]
  y <- train_data$team100_win
  
  # Bin the predictor into 10 groups
  bins <- cut(x, breaks = 10, include.lowest = TRUE)
  
  # For each bin, calculate the log-odds
  bin_means <- tapply(x, bins, mean)
  bin_logodds <- tapply(y, bins, function(p) {
    p_hat <- mean(p)
    p_hat <- pmax(pmin(p_hat, 0.999), 0.001)
    log(p_hat / (1 - p_hat))
  })
  
  plot(bin_means, bin_logodds,
       main = feat, cex.main = 0.7,
       xlab = "Predictor value (bin mean)",
       ylab = "Log-odds",
       pch = 19)
  abline(lm(bin_logodds ~ bin_means), col = "red")
}












# We dont want the model to overfit so we will check EPV
# and calculate how many features should we drop
n_events <- min(table(train_data$team100_win))
n_predictors <- length(current_predictors)
epv <- n_events / n_predictors

# our EPV (we aim for >= 10)
epv
# Max predictors we can afford
goal <- floor(n_events / 10)
goal
# we need to drop this many features
drop_count <- n_predictors - goal
drop_count


# Calculate absolute correlation with target
target_cor <- sapply(current_predictors, function(f) {
  abs(cor(train_data[[f]], train_data$team100_win, use = "complete.obs"))
})

# Sort from weakest to strongest
target_cor_sorted <- sort(target_cor)


# Drop the bottom N features
drop_weak <- names(target_cor_sorted)[1:drop_count]
remaining <- setdiff(current_predictors, drop_weak)
length(remaining)

# Check new EPV
round(n_events / length(remaining), 1)

# Apply
train_data <- train_data[, c("team100_win", remaining)]
test_data <- test_data[, c("team100_win", remaining)]

# Refit and check
model <- glm(team100_win ~ ., data = train_data, family = "binomial")










