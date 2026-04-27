library(shiny)
library(e1071)
library(ggplot2)

data <- read.csv2("processed.csv")
data <- data[, -1]
data$target <- as.factor(data$target)



all_features <- setdiff(names(data), "target")

feature_groups <- list(
  "Combat" = grep("kill|damage|spree|critical|assists", all_features, value = TRUE),
  "Vision" = grep("vision|ward|detector", all_features, value = TRUE),
  "Objectives" = grep("dragon|baron|inhibitor|tower|objective", all_features, value = TRUE),
  "Economy" = grep("gold|items|minion|jungle", all_features, value = TRUE),
  "Pings" = grep("ping", all_features, value = TRUE),
  "Other" = grep("time_played|cc_dealt|ccing_others|heal|units_healed|longest_time|summoner", all_features, value = TRUE)
)

ui <- fluidPage(
  theme = bslib::bs_theme(bootswatch = "darkly"),
  titlePanel("Naive Bayes Explorer - League of Legends"),
  
  sidebarLayout(
    sidebarPanel(
      width = 3,
      h4("Parameters"),
      sliderInput("laplace", "Laplace Smoothing:", min = 0, max = 10, value = 1, step = 0.5),
      sliderInput("train_split", "Train Split:", min = 0.5, max = 0.9, value = 0.8, step = 0.05),
      numericInput("seed", "Random Seed:", value = 123, min = 1),
      hr(),
      h4("Feature Selection"),
      actionButton("select_all", "Select All", class = "btn-sm btn-primary"),
      actionButton("select_none", "Deselect All", class = "btn-sm btn-danger"),
      br(), br(),
      lapply(names(feature_groups), function(group) {
        checkboxGroupInput(
          inputId = paste0("features_", group),
          label = paste0(group, " (", length(feature_groups[[group]]), ")"),
          choices = feature_groups[[group]],
          selected = feature_groups[[group]]
        )
      }),
      hr(),
      textOutput("feature_count")
    ),
    
    mainPanel(
      width = 9,
      h4("Model Performance"),
      tableOutput("metrics_table"),
      hr(),
      fluidRow(
        column(6, plotOutput("confusion_plot", height = "350px")),
        column(6, plotOutput("density_plot", height = "350px"))
      ),
      fluidRow(
        column(6, plotOutput("importance_plot", height = "400px")),
        column(6, plotOutput("roc_plot", height = "400px"))
      )
    )
  )
)

server <- function(input, output, session) {
  
  observeEvent(input$select_all, {
    lapply(names(feature_groups), function(group) {
      updateCheckboxGroupInput(session, paste0("features_", group),
                               selected = feature_groups[[group]])
    })
  })
  
  observeEvent(input$select_none, {
    lapply(names(feature_groups), function(group) {
      updateCheckboxGroupInput(session, paste0("features_", group),
                               selected = character(0))
    })
  })
  
  selected_features <- reactive({
    feats <- c()
    for (group in names(feature_groups)) {
      feats <- c(feats, input[[paste0("features_", group)]])
    }
    feats
  })
  
  output$feature_count <- renderText({
    paste0("Selected: ", length(selected_features()), " / ", length(all_features), " features")
  })
  
  nb_result <- reactive({
    req(length(selected_features()) > 0)
    set.seed(input$seed)
    
    idx <- sample(1:nrow(data), size = floor(input$train_split * nrow(data)))
    feats <- selected_features()
    
    train_x <- data[idx, feats, drop = FALSE]
    train_y <- data$target[idx]
    test_x <- data[-idx, feats, drop = FALSE]
    test_y <- data$target[-idx]
    
    model <- naiveBayes(x = train_x, y = train_y, laplace = input$laplace)
    
    pred_class <- predict(model, test_x)
    pred_prob <- predict(model, test_x, type = "raw")
    
    lvls <- levels(data$target)
    tp <- sum(pred_class == lvls[2] & test_y == lvls[2])
    tn <- sum(pred_class == lvls[1] & test_y == lvls[1])
    fp <- sum(pred_class == lvls[2] & test_y == lvls[1])
    fn <- sum(pred_class == lvls[1] & test_y == lvls[2])
    
    accuracy <- (tp + tn) / length(test_y)
    precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
    recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
    specificity <- ifelse(tn + fp > 0, tn / (tn + fp), 0)
    f1 <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
    
    list(
      model = model,
      predictions = pred_class,
      probabilities = pred_prob,
      test_y = test_y,
      tp = tp, tn = tn, fp = fp, fn = fn,
      accuracy = accuracy,
      precision = precision,
      recall = recall,
      specificity = specificity,
      f1 = f1,
      pos_class = lvls[2]
    )
  })
  
  output$metrics_table <- renderTable({
    req(nb_result())
    r <- nb_result()
    data.frame(
      Metric = c("Accuracy", "Precision", "Recall", "Specificity", "F1 Score"),
      Value = sprintf("%.4f", c(r$accuracy, r$precision, r$recall, r$specificity, r$f1))
    )
  }, striped = TRUE, hover = TRUE, bordered = TRUE)
  
  output$confusion_plot <- renderPlot({
    req(nb_result())
    r <- nb_result()
    lvls <- levels(data$target)
    
    cm_df <- data.frame(
      Prediction = factor(c(lvls[2], lvls[1], lvls[2], lvls[1]), levels = lvls),
      Actual = factor(c(lvls[2], lvls[2], lvls[1], lvls[1]), levels = lvls),
      Count = c(r$tp, r$fn, r$fp, r$tn)
    )
    
    ggplot(cm_df, aes(x = Actual, y = Prediction, fill = Count)) +
      geom_tile(color = "white", linewidth = 1.5) +
      geom_text(aes(label = Count), size = 8, fontface = "bold", color = "white") +
      scale_fill_gradient(low = "#1e293b", high = "#6366f1") +
      labs(title = "Confusion Matrix", x = "Actual", y = "Predicted") +
      theme_minimal(base_size = 14) +
      theme(
        plot.background = element_rect(fill = "#0f172a", color = NA),
        panel.background = element_rect(fill = "#0f172a", color = NA),
        text = element_text(color = "#e2e8f0"),
        axis.text = element_text(color = "#94a3b8"),
        legend.position = "none",
        plot.title = element_text(hjust = 0.5, face = "bold")
      )
  })
  
  output$density_plot <- renderPlot({
    req(nb_result())
    r <- nb_result()
    
    prob_df <- data.frame(
      probability = r$probabilities[, r$pos_class],
      actual = r$test_y
    )
    
    ggplot(prob_df, aes(x = probability, fill = actual)) +
      geom_density(alpha = 0.5) +
      geom_vline(xintercept = 0.5, linetype = "dashed", color = "#fbbf24", linewidth = 1) +
      scale_fill_manual(values = c("#fb7185", "#2dd4bf")) +
      labs(title = "Predicted Probability Distribution",
           x = paste0("P(", r$pos_class, ")"), y = "Density", fill = "Actual") +
      theme_minimal(base_size = 14) +
      theme(
        plot.background = element_rect(fill = "#0f172a", color = NA),
        panel.background = element_rect(fill = "#0f172a", color = NA),
        text = element_text(color = "#e2e8f0"),
        axis.text = element_text(color = "#94a3b8"),
        legend.background = element_rect(fill = "#1e293b"),
        legend.text = element_text(color = "#e2e8f0"),
        plot.title = element_text(hjust = 0.5, face = "bold")
      )
  })
  
  output$importance_plot <- renderPlot({
    req(nb_result())
    r <- nb_result()
    tables <- r$model$tables
    
    importance <- sapply(tables, function(tbl) {
      if (is.matrix(tbl) && ncol(tbl) >= 2) {
        abs(tbl[1, 1] - tbl[2, 1])
      } else {
        0
      }
    })
    
    imp_df <- data.frame(
      feature = names(importance),
      importance = as.numeric(importance)
    )
    imp_df$importance <- imp_df$importance / max(imp_df$importance)
    imp_df <- imp_df[order(-imp_df$importance), ]
    imp_df <- head(imp_df, 15)
    imp_df$feature <- gsub("diff_", "", imp_df$feature)
    imp_df$feature <- factor(imp_df$feature, levels = rev(imp_df$feature))
    
    ggplot(imp_df, aes(x = feature, y = importance, fill = importance)) +
      geom_col() +
      coord_flip() +
      scale_fill_gradient(low = "#6366f1", high = "#a78bfa") +
      labs(title = "Feature Importance (Top 15)", x = "", y = "Relative Importance") +
      theme_minimal(base_size = 14) +
      theme(
        plot.background = element_rect(fill = "#0f172a", color = NA),
        panel.background = element_rect(fill = "#0f172a", color = NA),
        text = element_text(color = "#e2e8f0"),
        axis.text = element_text(color = "#94a3b8"),
        legend.position = "none",
        plot.title = element_text(hjust = 0.5, face = "bold")
      )
  })
  
  output$roc_plot <- renderPlot({
    req(nb_result())
    r <- nb_result()
    
    prob_pos <- r$probabilities[, r$pos_class]
    actual_binary <- ifelse(r$test_y == r$pos_class, 1, 0)
    
    thresholds <- seq(0, 1, by = 0.01)
    roc_points <- data.frame(
      tpr = sapply(thresholds, function(t) {
        pred <- ifelse(prob_pos >= t, 1, 0)
        sum(pred == 1 & actual_binary == 1) / max(1, sum(actual_binary == 1))
      }),
      fpr = sapply(thresholds, function(t) {
        pred <- ifelse(prob_pos >= t, 1, 0)
        sum(pred == 1 & actual_binary == 0) / max(1, sum(actual_binary == 0))
      })
    )
    
    roc_sorted <- roc_points[order(roc_points$fpr, roc_points$tpr), ]
    auc <- abs(sum(diff(roc_sorted$fpr) * (head(roc_sorted$tpr, -1) + tail(roc_sorted$tpr, -1)) / 2))
    
    ggplot(roc_points, aes(x = fpr, y = tpr)) +
      geom_line(color = "#6366f1", linewidth = 1.5) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#475569") +
      annotate("text", x = 0.6, y = 0.3, label = paste0("AUC = ", round(auc, 4)),
               color = "#a78bfa", size = 5, fontface = "bold") +
      labs(title = "ROC Curve", x = "False Positive Rate", y = "True Positive Rate") +
      theme_minimal(base_size = 14) +
      theme(
        plot.background = element_rect(fill = "#0f172a", color = NA),
        panel.background = element_rect(fill = "#0f172a", color = NA),
        text = element_text(color = "#e2e8f0"),
        axis.text = element_text(color = "#94a3b8"),
        plot.title = element_text(hjust = 0.5, face = "bold")
      )
  })
}

shinyApp(ui = ui, server = server)
