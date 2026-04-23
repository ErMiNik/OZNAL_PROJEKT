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

# are scuwed
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

dim(wide_df)

# No need for match id from now on, drop it
wide_df$match_id <- NULL


# drop every feature that has >= 0.8 corelation 


cor_matrix <- cor(wide_df, use = "complete.obs")

library(caret)

cutoff <- 0.8

high_cor_names <- findCorrelation(cor_matrix, cutoff = cutoff, names = TRUE)

target <- wide_df$team100_win

results <- do.call(rbind, lapply(high_cor_names, function(col) {
  cors <- abs(cor_matrix[col, ])
  partners <- names(cors[cors >= cutoff & names(cors) != col])
  
  do.call(rbind, lapply(partners, function(p) {
    data.frame(
      col1 = col,
      col2 = p,
      cor_with_target_col1 = abs(cor(wide_df[[col]], target, use = "complete.obs")),
      cor_with_target_col2 = abs(cor(wide_df[[p]], target, use = "complete.obs"))
    )
  }))
}))

results$drop <- ifelse(results$cor_with_target_col1 < results$cor_with_target_col2,
                       results$col1, results$col2)
cols_to_drop <- unique(results$drop)
cols_to_drop
df_clean <- wide_df[, !(names(wide_df) %in% cols_to_drop)]

dim(df_clean)
write.csv2(df_clean, "processed.csv")
