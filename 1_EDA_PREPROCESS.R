# Dataset: https://huggingface.co/datasets/BoostedJonP/league_of_legends_match_data
# Goal: Binary classification (win/lose)
# based on end-of-game stats
# we will be modeling post-game results not pre-game prediction



# load EDA libraries

library(tidyverse)
library(magrittr)
library(patchwork)

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

# No need for match id from now on, drop it
wide_df$match_id <- NULL
# also drop team200_win we will use team100_win as our target
wide_df$team200_win <- NULL

predictors <- setdiff(names(wide_df), "team100_win")

# Check near zero variance again with aggregated data
nzv <- nearZeroVar(wide_df[, predictors], saveMetrics = TRUE)
drop_nzv <- rownames(nzv[nzv$nzv == TRUE, ])
print(drop_nzv)
remaining <- setdiff(predictors, drop_nzv)

wide_df <- wide_df[, !(names(wide_df) %in% drop_nzv)]



# define function for checking highly correlated features
get_high_cor <- function(df){
  
  # Lets look at highly corelated feature and drop any that seem redundant
  predictors <- setdiff(names(df), "team100_win")
  cor_matrix <- cor(df[, predictors], use = "complete.obs")
  
  # Get upper triangle pairs only (avoid duplicates)
  pairs <- which(upper.tri(cor_matrix), arr.ind = TRUE)
  
  cor_list <- data.frame(
    feature_1 = rownames(cor_matrix)[pairs[, 1]],
    feature_2 = colnames(cor_matrix)[pairs[, 2]],
    correlation = round(cor_matrix[pairs], 4),
    abs_correlation = round(abs(cor_matrix[pairs]), 4),
    stringsAsFactors = FALSE
  )
  
  # Sort by absolute correlation descending
  cor_list <- cor_list[order(cor_list$abs_correlation, decreasing = TRUE), ]
  
  # Show only |r| >= 0.8
  high_cor <- cor_list[cor_list$abs_correlation >= 0.8, ]
  return(high_cor)
}


high_cor <- get_high_cor(wide_df)
# game duration and time played - same numbers drop one of them
# team200_kills - team100_deaths 0.99 coralation
# team200_deaths - team100_kills - same problem
# we will only keep team100 stats that are redundant with team200
# also team100_[TYPE]_damage_dealt_to_champions -
# team200_[TYPE]_damage_taken

# team100_gold_earned - team100_gold_spent 0.98 corelation

high_cor_drop <- c("time_played", "team200_kills", "team200_deaths", "team200_assists",
                   "team200_champion_level", "team100_assists", "team100_gold_per_minute",
                   "team200_gold_per_minute",
                   
                   "team200_physical_damage_dealt",
                   "team200_physical_damage_dealt_to_champions",
                   "team200_physical_damage_taken",
                   
                   "team200_magic_damage_dealt",
                   "team200_magic_damage_dealt_to_champions",
                   "team200_magic_damage_taken",
                   
                   "team200_true_damage_dealt",
                   "team200_true_damage_dealt_to_champions",
                   "team200_true_damage_taken",
                   
                   "team200_total_damage_dealt",
                   "team200_total_damage_dealt_to_champions",
                   "team200_total_damage_taken",
                   
                   "team200_damage_self_mitigated",
                   
                   "team200_gold_earned", 'team100_gold_earned',
                   
                   "team200_first_blood_kill", "team200_first_tower_kill",
                   
                   "team100_champion_experience", "team200_champion_experience",
                   
                   "team200_vision_wards_bought_in_game",
                   
                   "team100_total_time_spent_dead", "team200_total_time_spent_dead",
                   
                   "team100_gold_spent", "team200_gold_spent",
                   
                   "team100_turret_takedowns", "team200_turret_takedowns",
                   
                   "team100_inhibitor_takedowns", "team200_inhibitor_takedowns",
                   "team100_inhibitor_kills", "team200_inhibitor_kills",
                   "team100_turret_kills", "team200_turret_kills",
                   
                   "team100_total_damage_dealt_to_champions",
                   
                   "team200_total_minions_killed",
                   
                   "team100_wards_placed", "team200_wards_placed",
                   "team100_detector_wards_placed", "team200_detector_wards_placed",
                   
                   "team100_items_purchased", "team200_items_purchased",
                   
                   "team100_wards_killed", "team200_wards_killed",
                   
                   "team100_vision_score", "team200_vision_score",
                   
                   "team200_total_ally_jungle_minions_killed",
                   
                   "team100_total_minions_killed", "team200_total_minions_killed",
                   "team100_neutral_minions_killed", "team200_neutral_minions_killed",
                   
                   "team200_largest_killing_spree", "team200_killing_sprees",
                   
                   "team100_total_damage_taken",
                   "team100_physical_damage_dealt_to_champions",
                   "team100_magic_damage_dealt_to_champions",
                   "team100_killing_sprees",
                   "team100_vision_wards_bought_in_game",
                   "team100_total_damage_dealt",
                   "team200_damage_per_minute",
                   "team100_damage_per_minute",
                   "team100_physical_damage_taken",
                   "game_duration")

# We iteratively run this code and removed highly correlated features (>= 0.8)
clean_wide <- wide_df[, !names(wide_df) %in% high_cor_drop]

high_cor2 <- get_high_cor(clean_wide)

dim(clean_wide)


# Lets try algorithmic approach that drops feature that has the most
# correlated features that are >= 0.8
predictors <- setdiff(names(wide_df), "team100_win")
remaining <- predictors
dropped <- c()

while (TRUE) {
  cor_matrix <- cor(wide_df[, remaining], use = "complete.obs")
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

clean_wide2 <- wide_df[, c("team100_win", remaining)]
dim(clean_wide2)



# Features the algorithm dropped that we didnt
setdiff(dropped, high_cor_drop)

# Features we dropped that algorithm didnt
setdiff(high_cor_drop, dropped)

length(dropped)
length(high_cor_drop)

# we still want to keep the most features so we choose the algorthmic
# approach
write.csv2(clean_wide2, "processed.csv")
