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

# ================ LOAD DATA ================
df <- read.csv2("processed.csv")
df <- df[, -1]  # drops the first column (index)

# check if data load was success
dim(df)


# ================ SPLIT ================
library(caret) #createDataPartition
set.seed(67) # replicability
index <- createDataPartition(df$target, p = 0.8, list = FALSE)
train_data <- df[index, ]
test_data  <- df[-index, ]



# ================ LOG-ODDS RELATIONSHIP ================

plot_logistic <- function(df, feature) {
  ggplot(df, aes(x = .data[[feature]], y = target)) +
    geom_point(alpha = 0.3) +
    geom_smooth(method = "glm", method.args = list(family = "binomial"), se = FALSE) +
    labs(x = feature, y = "P(target = 1)") +
    theme_minimal()
}

pdf("logistic_plots.pdf")
lapply(names(train_data), function(f) plot_logistic(train_data, f))
dev.off()

# We selected only features that copy sigmoid

predictors <- c("diff_assists",
                "diff_triple_kills",
                "diff_largest_multi_kill",
                "diff_killing_sprees",
                "diff_total_damage_dealt",
                "diff_total_damage_taken",
                "diff_physical_damage_dealt",
                "diff_true_damage_dealt",
                "diff_gold_spent",
                "diff_total_enemy_jungle_minions_killed",
                "diff_dragon_kills",
                "diff_baron_kills",
                "diff_inhibitor_takedowns",
                "diff_total_heal",
                "diff_solo_kills",
                "diff_outnumbered_kills"
)

train_data <- train_data[, c("target", predictors)]
test_data <- test_data[, c("target", predictors)]


# ================ VIF CHECK ================
# Iterative VIF removal
threshold <- 5
current_predictors <- setdiff(names(train_data), "target")
library(car) # model
repeat {
  current_formula <- as.formula(
    paste("target ~", paste(current_predictors, collapse = " + "))
  )
  current_model <- glm(current_formula, data = train_data, family = "binomial")
  
  vif_vals <- vif(current_model)
  max_vif <- max(vif_vals)
  
  if (max_vif <= threshold) break
  
  drop <- names(which.max(vif_vals))
  current_predictors <- setdiff(current_predictors, drop)
}

# Before VIF
length(predictors)
#After VIF cleanup
#Remaining features
length(current_predictors)
# Converged
current_model$converged

train_data <- train_data[, c("target", current_predictors)]
test_data <- test_data[, c("target", current_predictors)]

# ================ EXTREME OUTLIERS CHECK ================
# beyond 3*IQR is extreme
for (feat in names(train_data)) {
  x <- train_data[[feat]]
  Q1 <- quantile(x, 0.25)
  Q3 <- quantile(x, 0.75)
  IQR_val <- Q3 - Q1
  
  extreme <- which(x < Q1 - 3 * IQR_val | x > Q3 + 3 * IQR_val)
  
  if (length(extreme) > 0) {
    cat(sprintf("%-45s extreme outliers: %d (%.2f%%)\n",
                feat, length(extreme),
                length(extreme) / length(x) * 100))
  }
}

# Winsorize (cap at 3 standard deviations)
for (feat in names(train_data)) {
  x <- train_data[[feat]]
  mu <- mean(x)
  s <- sd(x)
  train_data[[feat]] <- pmax(pmin(x, mu + 3*s), mu - 3*s)
  
  # Apply same caps to test set using TRAIN mean/sd
  test_data[[feat]] <- pmax(pmin(test_data[[feat]], mu + 3*s), mu - 3*s)
}

# We dont have much to do with these outliers
# but we also dont count them as outliers
hist(train_data$diff_triple_kills)
boxplot(train_data$diff_triple_kills)


# ================ MODEL FIT ================
model <- glm(target ~ ., data = train_data, family = "binomial")
model
summary(model)
train_probs <- predict(model, type = "response")
mean((train_probs > 0.5) == train_data$target)

test_probs <- predict(model, newdata = test_data, type = "response")
mean((test_probs > 0.5) == test_data$target)






