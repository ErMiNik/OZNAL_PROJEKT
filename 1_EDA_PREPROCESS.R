# Dataset: https://huggingface.co/datasets/BoostedJonP/league_of_legends_match_data
# Goal: Binary classification (win/lose)
# based on end-of-game stats
# we will be modeling post-game results not pre-game prediction


# load EDA libraries

library(tidyverse)
library(magrittr)
library(patchwork)
library(caret)

# load dataset
df <- read_csv("data.csv")

# Basic exploritory of dataset

dim(df)

colnames(df)

str(df)

# we have a lot of useful numeric columns
# some are useless as champion_id
# but we will filter them later on

table(df$queue_id)

# we will only work with queue_id 420
# which is the soloQ ranked, this gamemode
# is more serious and competetive
# so the data will be more consistant
# it would be bad for the model to mix
# competive gamemode and for fun games

df %<>% filter(., queue_id==420)

length(unique(df$match_id))
# The dataset is in long format
# 1 row = 1 player
# in match soloQ match there are 10 players
# There are 2 teams each has 5 players

# We have 1216 unique matches, but there are
# 21360 rows
# each match should have 10 rows (5 players for each team)
# that means we probably have some duplicates


table(df$match_id)
# We can see a lot of duplicated rows
# because there should only be 10 match ids

any(duplicated(df))
# lets remove all duplicated rows

df <- df[!duplicated(df), ]
length(unique(df$match_id))
# the rows*10 and unique matches matches now
# we still need to check if ALL match ids are 
# in count of 10
table(table(df$match_id))
# we are good

# Lets check missing values
sum(is.na(df))

# we have 15 missing values, lets locate
# them and decide what to do

which(is.na(df), arr.ind = TRUE)
colnames(df)[10]

# The missing value is only team_position
# we dont have to worry about that for now
# we defenetaly dont want to drop rows
# that would mean losing 1 person for a
# 5 players team, which will disturb the 
# balance later on




# Count surrenders per match
surrender_count <- aggregate(
  game_ended_in_surrender ~ match_id,
  data = df,
  FUN = sum
)

cat("Matches with early surrender:\n")
print(table(surrender_count$game_ended_in_surrender))



# there are 315 matches that ended in surrender, we dont want to lose too many
# matches, but we will drop any matches that ended before the first blood kill
# and first blood tower later after aggregating the data
# this is because game that ended even before first tower or
# first blood is not normal and just noise in our data




# Data leakage
# our model will be predicting the result of which team
# wins the game
# This end of game statistic can be told by other variable
# we need to drop these columns to not intorduce
# data leakage to our model
# lets analyze the colmns and decide what to drop
colnames(df)
# - nexus_kills/nexus_takedowns - killing the nexus ends the game
# but team that kills the nexus can still surrender
# very rare edge case tho
# - turret a inhibitor takedowns/kills - these variable dont expicitly
# tells us who wins the game but are very powerfull

# - game_ended_in_(early)_surrender - tells us only that the game 
# ended in surrender, not which team surrendered
# - team_early_surrendered - dataleakage


# lets check these edge predictors agains target
edge_leakage <- c("win",
                  "nexus_kills",
                  "nexus_takedowns",
                  "turret_takedowns",
                  "turret_kills",
                  "inhibitor_takedowns",
                  "inhibitor_kills",
                  "game_ended_in_surrender",
                  "game_ended_in_early_surrender",
                  "team_early_surrendered")

for (col in edge_leakage) {
  cat("\n---", col, "---\n")
  print(table(df[[col]], df$win))
}


# From the tables we can tell few things
# nexus kills and takedowns are direct predictors
# if any > 0 that means the team won so we
# will drop these columns

# same with team_early_surrendered, if any team True that means
# the team lost, we will drop it

# game ended in (early) surrender carries 0 information
# about who won, so we can also just drop it

# we will check other variables after aggregating the rows
# we could only perform this table check

leakage_cols <- c("nexus_kills",
               "nexus_takedowns",
               "team_early_surrendered",
               "game_ended_in_surrender",
               "game_ended_in_early_surrender")


df <- df[, !(names(df) %in% leakage_cols)]


# Lets analyze the columns more and drop any that have 
# no meaning for us
colnames(df)
# - queue_id - no need for that anymore
# - champion_id/name - hard to implement for information
# - champion_skin_id - same as above
# - individual/team_position - no info
# - role/lane - no info
# - summoner_1/2_id - wont be able to use after aggregation
# - spell_x_casts - irrelevant after aggregation, also differ heavely for champ
# - eligible_for_progression - no info
# - all rune stats are useless
# - bans are also useless
drop_cols <- c("queue_id",
               "champion_id",
               "champion_name",
               "champion_skin_id",
               "team_position",
               "individual_position",
               "role",
               "lane",
               "summoner_1_id",
               "summoner_2_id",
               "spell_1_casts",
               "spell_2_casts",
               "spell_3_casts",
               "spell_4_casts",
               "eligible_for_progression",
               "rune_primary_style",
               "rune_secondary_style",
               "rune_stat_defense",
               "rune_stat_flex",
               "rune_stat_offense",
               "ban_1_champion_id",
               "ban_2_champion_id",
               "ban_3_champion_id",
               "ban_4_champion_id",
               "ban_5_champion_id")

df <- df[, !(names(df) %in% drop_cols)]


# we will also check cols with zero variance and decide which to drop

zero_var <- sapply(df, function(x) var(x, na.rm = TRUE))
zero_var[zero_var == 0]
# unreal kills are not in the game for a few years, this is irrelevant for us
# sight wards bought in game is weird that noone bought it but we can drop it
# effective heal and shielding
# basic, danger, hold and vision cleared pings are also not used
# we can drop them all

cols_to_drop <- names(zero_var[zero_var == 0])
df <- df[, !names(df) %in% cols_to_drop]


# Univariate analysis

summary(df)

nums <- names(df)[sapply(df, is.numeric)]

par(mfrow = c(2, 2), mar = c(2, 2, 2, 1))

for (i in seq_along(nums)) {
  hist(df[[nums[i]]], main = nums[i], xlab = "")
  
  if (i %% 4 == 0 && i < length(nums)) {
    par(mfrow = c(2, 2))
  }
}

par(mfrow = c(1, 1))

# Observations from histograms
# looks noramlly distributed
# - game duration
# - champion experience
# - gold earned
# - gold spent
# - items purchased
# - time played
# - kill participation

# are scewed
# - champ lvl
# - kills
# - deaths
# - assists
# - double kills
# - Largest killing spree
# - killing sprees
# - all dmg taken/dealt
# - consumables purchased
# - vision score
# - wards stats
# - turret takedowns
# - inhibitor takedowns
# - total time spent dead
# - total time cc dealt
# - time accing others
# - total heal
# - largest crit strike
# - ping stats
# - KDA
# - solo kills
# - outnumbered kills




par(mfrow = c(2, 2), mar = c(2, 2, 2, 1))

for (i in seq_along(nums)) {
  boxplot(df[[nums[i]]], main = nums[i], xlab = "")
  
  if (i %% 4 == 0 && i < length(nums)) {
    par(mfrow = c(2, 2))
  }
}

par(mfrow = c(1, 1))



detect_outliers <- function(df) {
  nums <- names(df)[sapply(df, is.numeric)]
  
  results <- data.frame(
    variable = character(),
    total = integer(),
    n_outliers = integer(),
    pct_outliers = numeric(),
    lower_bound = numeric(),
    upper_bound = numeric(),
    min_value = numeric(),
    max_value = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (col in nums) {
    x <- df[[col]][!is.na(df[[col]])]
    
    q1 <- quantile(x, 0.25)
    q3 <- quantile(x, 0.75)
    iqr <- q3 - q1
    
    lower <- q1 - 1.5 * iqr
    upper <- q3 + 1.5 * iqr
    
    outliers <- x[x < lower | x > upper]
    
    results <- rbind(results, data.frame(
      variable = col,
      n_outliers = length(outliers)
    ))
  }
  
  results <- results[order(-results$n_outliers), ]
  rownames(results) <- NULL
  return(results)
}

outlier_summary <- detect_outliers(df)

# see only variables with outliers
outlier_summary[outlier_summary$n_outliers > 0, ]

# now that we have almost clean data, last thing we need to do
# is to convert logical (TRUE/FALSE) datatypes to bool
# the only chr type we have left is match_id which we will
# use to group the rows together

logic_cols <- names(df)[sapply(df, is.logical)]
logic_cols

# that is
# - win (our target)
# - first tower/blood kill/assist
# Lets convert them

df %<>% mutate(across(where(is.logical), as.numeric))

# last sanity check that our data is correct

# each team in match should have 5 players 
df %>% count(match_id, team_id) %>% filter(n != 5)
# each match should have 2 teams
df %>% group_by(match_id) %>% 
  summarise(n_teams = n_distinct(team_id)) %>% 
  filter(n_teams != 2)
# each team in match should have one win value
df %>% group_by(match_id, team_id) %>% 
  summarise(n_win_vals = n_distinct(win), .groups="drop") %>% 
  filter(n_win_vals != 1)
# each match should have one winning and one loosing team
df %>% group_by(match_id) %>% 
  summarise(n_win_vals = n_distinct(win)) %>%
  filter(n_win_vals != 2)


# ==================== AGREGATE ==================== 

first_cols <- c("game_duration", "time_played", "win")
mean_cols <- c("kda", "kill_participation", "gold_per_minute",
               "vision_score_per_minute", "damage_per_minute")

sum_cols <- setdiff(names(df), 
                    c(first_cols, mean_cols,
                      "match_id", "team_id"))

# .names keeps the original name, without it we would get kda_mean or kills_sum
team_agg <- df %>%
  group_by(match_id, team_id) %>%
  summarise(
    across(all_of(first_cols), first, .names="{.col}"),
    across(all_of(mean_cols), mean, .names="{.col}"),
    across(all_of(sum_cols), sum, .names="{.col}"),
    .groups = "drop"
  )

# get all columns that should be pivoted (everything except id cols)
value_cols <- setdiff(names(team_agg), c("match_id", "team_id", "team", "game_duration", "time_played"))

wide_df <- team_agg %>%
  group_by(match_id) %>%
  mutate(team = paste0("team", team_id)) %>%
  ungroup() %>%
  pivot_wider(
    id_cols = c(match_id, game_duration, time_played),
    names_from = team,
    values_from = all_of(value_cols),
    names_glue = "{team}_{.value}"
  )

target <- wide_df$team100_win

# After pivot_wider, compute diffs for all value_cols
diff_df <- wide_df %>%
  mutate(across(
    starts_with("team100_"),
    ~ . - get(sub("team100_", "team200_", cur_column())),
    .names = "{sub('team100_', 'diff_', .col)}"
  )) %>%
  select(match_id, game_duration, time_played, starts_with("diff_"))

diff_df$target <- target
diff_df$match_id <- NULL
diff_df$diff_win <- NULL




# As we discussed above, drop any noise game that ended before tower kill or first blood kill

cat("First blood = 0:", sum(diff_df$diff_first_blood_kill == 0), "\n")

# Remove them
diff_df <- diff_df[diff_df$diff_first_blood_kill != 0, ]


# Now the values should only be 1 and -1
print(table(diff_df$diff_first_blood_kill))


cat("First tower = 0:", sum(diff_df$diff_first_tower_kill == 0), "\n")

# If there are zeros, remove them too
diff_df <- diff_df[diff_df$diff_first_tower_kill != 0, ]

print(table(diff_df$diff_first_tower_kill))

cat("Rows remaining:", nrow(diff_df), "\n")





predictors <- setdiff(names(diff_df), "target")

# ==================== ZERO VAR CHECK ==================== 

# Check near zero variance again with aggregated data
nzv <- nearZeroVar(diff_df[, predictors], saveMetrics = TRUE)
drop_nzv <- rownames(nzv[nzv$nzv == TRUE, ])
print(drop_nzv)
remaining <- setdiff(predictors, drop_nzv)

diff_df <- diff_df[, !(names(diff_df) %in% drop_nzv)]

dim(diff_df)


# ==================== RESEARCH QUESTIONS ====================
# RQ1: 
# Is there a significant difference in the pings differential and their 
# type between winning and losing teams?




offensive_pings <- c("diff_push_pings", "diff_on_my_way_pings", "diff_assist_me_pings", "diff_all_in_pings")
defensive_pings <- c("diff_retreat_pings", "diff_get_back_pings", "diff_enemy_vision_pings", "diff_enemy_missing_pings")

all_pings <- grep("ping", names(diff_df), value = TRUE)
all_pings <- rowSums(diff_df[, all_pings])
offensive_ping_diff <- rowSums(diff_df[, offensive_pings])
defensive_ping_diff <- rowSums(diff_df[, defensive_pings])
hypothesis_df <- data.frame(offensive_ping_diff, defensive_ping_diff, all_pings, target = diff_df$target)

# p = 0.05
# H0: The mean offensive ping differential is the same for winning and losing teams
# H1: The mean offensive ping differential differs between winning and losing teams


library(moments)
shapiro.test(hypothesis_df$offensive_ping_diff)
# using Mann-Whitney U test because data is not normally distributed
wilcox.test(offensive_ping_diff ~ target, data = hypothesis_df)

# p < 0.05 Rejecting H0


# p = 0.05
# H0: The mean defensive ping differential is the same for winning and losing teams
# H1: The mean defensive ping differential differs between winning and losing teams

shapiro.test(hypothesis_df$defensive_ping_diff)
# using Mann-Whitney U test because data is not normally distributed
wilcox.test(defensive_ping_diff ~ target, data = hypothesis_df)

# p < 0.05 Rejecting H0


# p = 0.05
# H0: The mean of all pings differential is the same for winning and losing teams
# H1: The mean of all pings differential differs between winning and losing teams

shapiro.test(hypothesis_df$all_pings)
# using Mann-Whitney U test because data is not normally distributed
wilcox.test(all_pings ~ target, data = hypothesis_df)

# p < 0.05 Rejecting H0


ping_long <- data.frame(
  outcome = rep(as.factor(hypothesis_df$target), 3),
  ping_type = rep(c("Offensive", "Defensive", "All"), each = nrow(hypothesis_df)),
  value = c(hypothesis_df$offensive_ping_diff, hypothesis_df$defensive_ping_diff, hypothesis_df$all_pings)
)

ggplot(ping_long, aes(x = outcome, y = value, fill = outcome)) +
  geom_boxplot() +
  facet_wrap(~ping_type, scales = "free_y") +
  scale_fill_manual(values = c("0" = "red", "1" = "blue"),
                    labels = c("Loss", "Win")) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  theme_minimal() +
  labs(title = "Ping Differential: Winners vs Losers",
       x = "Match Outcome",
       y = "Ping Difference (team100 - team200)",
       fill = "Outcome")

cat("=== Mean Ping Differentials ===\n\n")
cat("Offensive pings:\n")
cat("  Winners:", mean(hypothesis_df$offensive_ping_diff[hypothesis_df$target == 1]), "\n")
cat("  Losers: ", mean(hypothesis_df$offensive_ping_diff[hypothesis_df$target == 0]), "\n")

cat("\nDefensive pings:\n")
cat("  Winners:", mean(hypothesis_df$defensive_ping_diff[hypothesis_df$target == 1]), "\n")
cat("  Losers: ", mean(hypothesis_df$defensive_ping_diff[hypothesis_df$target == 0]), "\n")

cat("\nAll pings:\n")
cat("  Winners:", mean(hypothesis_df$all_pings[hypothesis_df$target == 1]), "\n")
cat("  Losers: ", mean(hypothesis_df$all_pings[hypothesis_df$target == 0]), "\n")


# Interpretation

# Winning teams use significantly more offensive pings 
# than their opponents, while losing teams use significantly more defensive 
# pings. This suggests that proactive communication 
# (push, all-in, on my way) is associated with winning, while reactive 
# communication (retreat, get back, assist me) is a sign of a losing team 
# that is constantly responding to the enemy's pressure. The total ping 
# differential is positive for winners indicating that winning teams
# communicate more overall, but this effect is primarily driven by offensive 
# ping usage.



# R2:
# Does early game advantage first blood + first tower
# significantly impact the result of the game ?




cat("First blood values:\n")
print(table(diff_df$diff_first_blood_kill))

cat("\nFirst tower values:\n")
print(table(diff_df$diff_first_tower_kill))


cat("\n=== First Blood ===\n")
print(table(diff_df$diff_first_blood_kill, diff_df$target))

cat("\n=== First Tower ===\n")
print(table(diff_df$diff_first_tower_kill, diff_df$target))


# H0: First blood and match outcome are independent
# H1: First blood and match outcome are not independent
# p = 0.05

cat("\n=== Chi-Square Test: First Blood ===\n")
fb_table <- table(diff_df$diff_first_blood_kill, diff_df$target)
chisq.test(fb_table)

# p > 0.05 cannot reject H0




# H0: First tower and match outcome are independent
# H1: First tower and match outcome are not independent

cat("\n=== Chi-Square Test: First Tower ===\n")
ft_table <- table(diff_df$diff_first_tower_kill, diff_df$target)
chisq.test(ft_table)

# p < 0.05 we reject H0, tower kill and match outcome are significantly depended


# ============================================
# Combined: Early game advantage
# ============================================
# Create combined variable
diff_df$early_advantage <- diff_df$diff_first_blood_kill + diff_df$diff_first_tower_kill

cat("\n=== Early Advantage Distribution ===\n")
print(table(diff_df$early_advantage, diff_df$target))

cat("\n=== Chi-Square Test: Early Advantage ===\n")
ea_table <- table(diff_df$early_advantage, diff_df$target)
chisq.test(ea_table)


cat("\n=== Win Rates ===\n")
win_rates <- tapply(diff_df$target, diff_df$early_advantage, mean)
cat("Win rate by early advantage score:\n")
print(round(win_rates, 4))




# Interpretation
# First blood alone does not significantly predict match outcome.
# However, first tower is a highly significant predictor.
# The combined early game advantage shows a clear dose-response
# pattern: teams that secured neither objective
# won only 28.6% of games, teams that split objectives won 49.5%, and teams
# that secured both won 70.8%. This suggests that first tower is the key
# early game objective, while first blood provides a smaller, non-significant
# advantage on its own.


comparison_df <- data.frame(
  objective = c("First Blood", "First Blood", "First Tower", "First Tower"),
  got_it = c("Yes", "No", "Yes", "No"),
  win_rate = c(
    mean(diff_df$target[diff_df$diff_first_blood_kill == 1]),
    mean(diff_df$target[diff_df$diff_first_blood_kill == -1]),
    mean(diff_df$target[diff_df$diff_first_tower_kill == 1]),
    mean(diff_df$target[diff_df$diff_first_tower_kill == -1])
  )
)

ggplot(comparison_df, aes(x = objective, y = win_rate, fill = got_it)) +
  geom_col(position = "dodge") +
  geom_text(aes(label = paste0(round(win_rate * 100, 1), "%")),
            position = position_dodge(width = 0.9), vjust = -0.5, size = 4) +
  scale_fill_manual(values = c("No" = "red", "Yes" = "blue")) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey50") +
  scale_y_continuous(labels = scales::percent, limits = c(0, 0.8)) +
  theme_minimal() +
  labs(title = "First Blood vs First Tower: Impact on Win Rate",
       x = "Objective", y = "Win Rate", fill = "Secured?")









clean_wide <- diff_df
# ==================== HIGH CORRELATION ==================== 

# define function for checking highly correlated features
get_high_cor <- function(df){
  
  # Lets look at highly corelated feature and drop any that seem redundant
  predictors <- setdiff(names(df), "target")
  cor_matrix <- cor(df[, predictors], use = "complete.obs")
  
  # Correlation of each predictor with target
  target_cor <- cor(df[, predictors], df$target, use = "complete.obs")[, 1]
  
  # Get upper triangle pairs only (avoid duplicates)
  pairs <- which(upper.tri(cor_matrix), arr.ind = TRUE)
  
  cor_list <- data.frame(
    feature_1 = rownames(cor_matrix)[pairs[, 1]],
    feature_2 = colnames(cor_matrix)[pairs[, 2]],
    correlation = round(cor_matrix[pairs], 4),
    abs_correlation = round(abs(cor_matrix[pairs]), 4),
    target_cor_1 = round(target_cor[rownames(cor_matrix)[pairs[, 1]]], 4),
    target_cor_2 = round(target_cor[colnames(cor_matrix)[pairs[, 2]]], 4),
    stringsAsFactors = FALSE
  )
  
  # Sort by absolute correlation descending
  cor_list <- cor_list[order(cor_list$abs_correlation, decreasing = TRUE), ]
  
  # Show only |r| >= 0.8
  high_cor <- cor_list[cor_list$abs_correlation >= 0.8, ]
  return(high_cor)
}


high_cor <- get_high_cor(diff_df)


# Lets try algorithmic approach that drops feature that has the most
# correlated features that are >= 0.8
predictors <- setdiff(names(diff_df), "target")
remaining <- predictors
dropped <- c()

while (TRUE) {
  cor_matrix <- cor(diff_df[, remaining], use = "complete.obs")
  diag(cor_matrix) <- 0
  
  # Count how many features each feature correlates with at |r| >= 0.8
  high_cor_count <- sapply(remaining, function(f) {
    sum(abs(cor_matrix[f, ]) >= 0.8)
  })
  
  # Stop if no more high correlations
  if (max(high_cor_count) == 0) break
  
  # Drop the feature with the most high correlations
  worst <- names(which.max(high_cor_count))
  
  dropped <- c(dropped, worst)
  remaining <- setdiff(remaining, worst)
}

# Features that we dropped and their correlation to target
cor(diff_df[, dropped], diff_df$target, use = "complete.obs")[, 1]

clean_wide2 <- diff_df[, c("target", remaining)]

dim(clean_wide2)

length(dropped)

write.csv2(clean_wide2, "processed.csv")
