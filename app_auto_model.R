packages <- c("shiny", "ggplot2", "glmnet", "pROC", "rpart", "rpart.plot")

install_if_missing <- function(pkgs) {
  missing <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
  if (length(missing) > 0) {
    cat("Installing:", paste(missing, collapse = ", "), "\n")
    install.packages(missing)
  }
}

install_if_missing(packages)

library(shiny)
library(ggplot2)
library(glmnet)
library(pROC)
library(rpart)
library(rpart.plot)

# Global variables
MANUAL_PRESET <- c(
  "diff_champion_experience",
  "diff_largest_multi_kill",
  "diff_dragon_kills",
  "diff_solo_kills"
)

DEFAULT_SEED <- 67

# UI section
ui <- fluidPage(
  titlePanel("Classification Model Explorer"),
  
  sidebarLayout(
    sidebarPanel(
      width = 3,
      
      # Model selector
      h4("Model"),
      selectInput(
        "model_type",
        NULL,
        choices = c(
          "Logistic Regression" = "logistic",
          "Decision Tree" = "tree"
        ),
        selected = "logistic"
      ),
      
      # Shared settings
      h4("Settings"),
      helpText("Best model was trained with seed 67", style = "color: orange;"),
      numericInput("seed", "Random Seed", value = DEFAULT_SEED, min = 1, max = 9999),
      sliderInput("split_ratio", "Train/Test Split", min = 0.5, max = 0.9, value = 0.7, step = 0.05),
      
      # Logistic Regression: Feature selection presets
      conditionalPanel(
        condition = "input.model_type == 'logistic'",
        hr(),
        h4("Feature Selection Presets"),
        actionButton("preset_manual", "My Selected Features", width = "100%"),
        actionButton("preset_lasso", "Lasso Selection", width = "100%"),
        actionButton("preset_stepwise", "Mixed Stepwise", width = "100%"),
        actionButton("preset_elasticnet", "Elastic Net Selection", width = "100%"),
        
        conditionalPanel(
          condition = "input.preset_elasticnet > 0",
          sliderInput("alpha", "Elastic Net Alpha", min = 0.05, max = 0.95, value = 0.5, step = 0.05)
        )
      ),
      
      # Decision Tree: Hyperparameters
      conditionalPanel(
        condition = "input.model_type == 'tree'",
        hr(),
        h4("Decision Tree Hyperparameters"),
        sliderInput("dt_maxdepth", "Max Depth", min = 1, max = 30, value = 10, step = 1),
        sliderInput("dt_minsplit", "Min Split (min obs to attempt split)", min = 2, max = 100, value = 20, step = 1),
        sliderInput("dt_minbucket", "Min Bucket (min obs in leaf)", min = 1, max = 50, value = 7, step = 1),
        sliderInput("dt_cp", "Complexity Parameter (cp)", min = 0.001, max = 0.1, value = 0.01, step = 0.001),
        helpText("Lower cp = more complex tree. Higher cp = more pruning.")
      ),
      
      # Features
      hr(),
      h4("Features"),
      textOutput("feature_count"),
      fluidRow(
        column(6, actionButton("select_all_features", "Select All Features", width = "100%")),
        column(6, actionButton("clear_all_features", "Clear All", width = "100%"))
      ),
      br(),
      div(
        style = "max-height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 5px;",
        uiOutput("feature_checkboxes")
      ),
      
      # Threshold
      hr(),
      h4("Classification Threshold"),
      sliderInput("threshold", NULL, min = 0.05, max = 0.95, value = 0.5, step = 0.01),
      actionButton("btn_youden", "Calculate Youden's Index", width = "100%")
    ),
    
    mainPanel(
      width = 9,
      tabsetPanel(
        id = "main_tabs",
        tabPanel("Metrics", uiOutput("metrics_display")),
        tabPanel("Confusion Matrix", plotOutput("confusion_plot", height = "400px")),
        tabPanel("ROC Curve", plotOutput("roc_plot", height = "450px")),
        
        # Logistic Regression specific tabs
        tabPanel("Coefficients", plotOutput("coef_plot", height = "500px")),
        tabPanel("Selection Summary", tableOutput("summary_table")),
        
        # Decision Tree specific tabs
        tabPanel("Tree Plot", plotOutput("tree_plot", height = "600px")),
        tabPanel("Variable Importance", plotOutput("varimp_plot", height = "450px")),
        tabPanel("CP Table", tableOutput("cp_table"))
      )
    )
  )
)

# Server section
server <- function(input, output, session) {
  
  # Load and split data
  data_split <- reactive({
    set.seed(input$seed)
    df <- read.csv2("processed.csv", stringsAsFactors = FALSE, row.names = 1)
    df$target <- as.factor(df$target)
    
    idx <- sample(1:nrow(df), size = floor(input$split_ratio * nrow(df)))
    train <- df[idx, ]
    test <- df[-idx, ]
    features <- setdiff(names(df), "target")
    
    list(train = train, test = test, features = features)
  })
  
  # Feature checkboxes
  output$feature_checkboxes <- renderUI({
    feats <- data_split()$features
    
    current_selection <- isolate(input$selected_features)
    current_selection <- current_selection[current_selection %in% feats]
    
    checkboxGroupInput(
      "selected_features",
      NULL,
      choices = feats,
      selected = current_selection
    )
  })
  
  output$feature_count <- renderText({
    n <- length(input$selected_features)
    paste0(n, " feature(s) selected")
  })
  
  observeEvent(input$select_all_features, {
    feats <- data_split()$features
    updateCheckboxGroupInput(session, "selected_features", selected = feats)
  })
  
  observeEvent(input$clear_all_features, {
    updateCheckboxGroupInput(session, "selected_features", selected = character(0))
  })
  
  # Presets, logistic only
  observeEvent(input$preset_manual, {
    feats <- data_split()$features
    selected <- MANUAL_PRESET[MANUAL_PRESET %in% feats]
    updateCheckboxGroupInput(session, "selected_features", selected = selected)
  })
  
  observeEvent(input$preset_lasso, {
    d <- data_split()
    x <- as.matrix(d$train[, d$features])
    y <- as.numeric(as.character(d$train$target))
    
    withProgress(message = "Running Lasso...", {
      cv_model <- cv.glmnet(x, y, alpha = 1, family = "binomial")
      coefs <- coef(cv_model, s = "lambda.min")
      selected <- rownames(coefs)[which(coefs != 0)][-1]
    })
    
    updateCheckboxGroupInput(session, "selected_features", selected = selected)
    showNotification(paste("Lasso selected", length(selected), "features"), type = "message")
  })
  
  observeEvent(input$preset_stepwise, {
    d <- data_split()
    
    withProgress(message = "Running Stepwise...", {
      null_model <- glm(target ~ 1, data = d$train, family = "binomial")
      step_model <- tryCatch(
        step(
          null_model,
          scope = list(
            lower = ~ 1,
            upper = formula(paste("~", paste(d$features, collapse = " + ")))
          ),
          direction = "both",
          trace = 0,
          control = glm.control(maxit = 100)
        ),
        error = function(e) NULL
      )
    })
    
    if (!is.null(step_model)) {
      selected <- names(coef(step_model))[-1]
      updateCheckboxGroupInput(session, "selected_features", selected = selected)
      showNotification(paste("Stepwise selected", length(selected), "features"), type = "message")
    } else {
      showNotification("Stepwise failed to converge", type = "error")
    }
  })
  
  observeEvent(input$preset_elasticnet, {
    d <- data_split()
    alpha_val <- input$alpha
    x <- as.matrix(d$train[, d$features])
    y <- as.numeric(as.character(d$train$target))
    
    withProgress(message = paste0("Running Elastic Net alpha=", alpha_val, "..."), {
      cv_model <- cv.glmnet(x, y, alpha = alpha_val, family = "binomial")
      coefs <- coef(cv_model, s = "lambda.min")
      selected <- rownames(coefs)[which(coefs != 0)][-1]
    })
    
    updateCheckboxGroupInput(session, "selected_features", selected = selected)
    showNotification(paste("Elastic Net selected", length(selected), "features"), type = "message")
  })
  
  # Automatic model fitting
  model_results <- reactive({
    feats <- input$selected_features
    
    validate(
      need(length(feats) > 0, "Select at least one feature to fit the model.")
    )
    
    d <- data_split()
    fml <- as.formula(paste("target ~", paste(feats, collapse = " + ")))
    
    if (input$model_type == "logistic") {
      model <- tryCatch(
        suppressWarnings(glm(fml, data = d$train, family = "binomial")),
        error = function(e) NULL
      )
      
      validate(
        need(!is.null(model), "Logistic Regression failed to fit.")
      )
      
      train_prob <- predict(model, type = "response")
      test_prob <- predict(model, newdata = d$test, type = "response")
      
    } else {
      model <- tryCatch(
        rpart(
          fml,
          data = d$train,
          method = "class",
          control = rpart.control(
            maxdepth = input$dt_maxdepth,
            minsplit = input$dt_minsplit,
            minbucket = input$dt_minbucket,
            cp = input$dt_cp
          )
        ),
        error = function(e) NULL
      )
      
      validate(
        need(!is.null(model), "Decision Tree failed to fit.")
      )
      
      train_prob <- predict(model, type = "prob")[, "1"]
      test_prob <- predict(model, newdata = d$test, type = "prob")[, "1"]
    }
    
    train_roc <- roc(d$train$target, train_prob, quiet = TRUE)
    test_roc <- roc(d$test$target, test_prob, quiet = TRUE)
    
    list(
      model = model,
      model_type = input$model_type,
      train = d$train,
      test = d$test,
      train_prob = train_prob,
      test_prob = test_prob,
      train_roc = train_roc,
      test_roc = test_roc,
      features = feats
    )
  })
  
  # Youden's Index
  observeEvent(input$btn_youden, {
    if (length(input$selected_features) == 0) {
      showNotification("Select at least one feature first", type = "error")
      return()
    }
    
    res <- model_results()
    
    best <- coords(res$test_roc, "best", best.method = "youden", ret = "threshold")
    best_val <- round(as.numeric(best), 2)
    best_val <- max(0.05, min(0.95, best_val))
    
    updateSliderInput(session, "threshold", value = best_val)
    showNotification(paste0("Youden's optimal threshold: ", best_val), type = "message")
  })
  
  # Helper: compute metrics
  compute_metrics <- function(prob, actual, threshold) {
    pred <- factor(ifelse(prob > threshold, 1, 0), levels = c("0", "1"))
    actual <- factor(actual, levels = c("0", "1"))
    tab <- table(Predicted = pred, Actual = actual)
    
    tp <- tab["1", "1"]
    tn <- tab["0", "0"]
    fp <- tab["1", "0"]
    fn <- tab["0", "1"]
    
    accuracy <- (tp + tn) / sum(tab)
    precision <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
    recall <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
    specificity <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
    f1 <- ifelse((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
    
    list(
      tab = tab,
      accuracy = accuracy,
      precision = precision,
      recall = recall,
      specificity = specificity,
      f1 = f1
    )
  }
  
  # Tab: Metrics
  output$metrics_display <- renderUI({
    res <- model_results()
    
    thr <- input$threshold
    train_m <- compute_metrics(res$train_prob, res$train$target, thr)
    test_m <- compute_metrics(res$test_prob, res$test$target, thr)
    train_auc <- round(as.numeric(auc(res$train_roc)), 4)
    test_auc <- round(as.numeric(auc(res$test_roc)), 4)
    auc_diff <- round(train_auc - test_auc, 4)
    
    fmt <- function(x) round(x, 4)
    
    model_label <- if (res$model_type == "logistic") "Logistic Regression" else "Decision Tree"
    
    overfit_msg <- if (auc_diff > 0.05) {
      p(
        style = "color: red; font-weight: bold;",
        paste0("Warning: Train-Test AUC gap is ", auc_diff, " - possible overfitting.")
      )
    } else {
      p(
        style = "color: green;",
        paste0("Train-Test AUC gap is ", auc_diff, " - no significant overfitting.")
      )
    }
    
    tagList(
      h3(paste0(model_label, "  |  Threshold: ", thr, "  |  Features: ", length(res$features))),
      hr(),
      
      h4("Test Set Metrics"),
      tags$table(
        style = "width: 60%; border-collapse: collapse;",
        tags$tr(
          tags$th("Metric", style = "text-align:left; padding:8px; border-bottom:2px solid #ddd;"),
          tags$th("Test", style = "text-align:right; padding:8px; border-bottom:2px solid #ddd;"),
          tags$th("Train", style = "text-align:right; padding:8px; border-bottom:2px solid #ddd;")
        ),
        tags$tr(
          tags$td("Accuracy", style = "padding:8px;"),
          tags$td(fmt(test_m$accuracy), style = "text-align:right; padding:8px;"),
          tags$td(fmt(train_m$accuracy), style = "text-align:right; padding:8px; color:gray;")
        ),
        tags$tr(
          tags$td("Precision", style = "padding:8px;"),
          tags$td(fmt(test_m$precision), style = "text-align:right; padding:8px;"),
          tags$td(fmt(train_m$precision), style = "text-align:right; padding:8px; color:gray;")
        ),
        tags$tr(
          tags$td("Recall", style = "padding:8px;"),
          tags$td(fmt(test_m$recall), style = "text-align:right; padding:8px;"),
          tags$td(fmt(train_m$recall), style = "text-align:right; padding:8px; color:gray;")
        ),
        tags$tr(
          tags$td("Specificity", style = "padding:8px;"),
          tags$td(fmt(test_m$specificity), style = "text-align:right; padding:8px;"),
          tags$td(fmt(train_m$specificity), style = "text-align:right; padding:8px; color:gray;")
        ),
        tags$tr(
          tags$td("F1 Score", style = "padding:8px;"),
          tags$td(fmt(test_m$f1), style = "text-align:right; padding:8px;"),
          tags$td(fmt(train_m$f1), style = "text-align:right; padding:8px; color:gray;")
        ),
        tags$tr(
          tags$td("AUC", style = "padding:8px; border-top:1px solid #ddd;"),
          tags$td(test_auc, style = "text-align:right; padding:8px; border-top:1px solid #ddd;"),
          tags$td(train_auc, style = "text-align:right; padding:8px; border-top:1px solid #ddd; color:gray;")
        )
      ),
      hr(),
      overfit_msg
    )
  })
  
  # Tab: Confusion Matrix
  output$confusion_plot <- renderPlot({
    res <- model_results()
    
    test_m <- compute_metrics(res$test_prob, res$test$target, input$threshold)
    tab_df <- as.data.frame(test_m$tab)
    tab_df$Label <- paste0(
      tab_df$Freq,
      "\n",
      ifelse(
        tab_df$Predicted == "1" & tab_df$Actual == "1", "(TP)",
        ifelse(
          tab_df$Predicted == "0" & tab_df$Actual == "0", "(TN)",
          ifelse(tab_df$Predicted == "1" & tab_df$Actual == "0", "(FP)", "(FN)")
        )
      )
    )
    
    ggplot(tab_df, aes(x = Actual, y = Predicted, fill = Freq)) +
      geom_tile(color = "white", linewidth = 2) +
      geom_text(aes(label = Label), size = 7, fontface = "bold") +
      scale_fill_gradient(low = "lightblue", high = "darkblue") +
      labs(title = "Test Confusion Matrix", x = "Actual", y = "Predicted") +
      theme_minimal(base_size = 14)
  })
  
  # Tab: ROC Curve
  output$roc_plot <- renderPlot({
    res <- model_results()
    
    test_df <- data.frame(
      sensitivity = res$test_roc$sensitivities,
      specificity = res$test_roc$specificities
    )
    train_df <- data.frame(
      sensitivity = res$train_roc$sensitivities,
      specificity = res$train_roc$specificities
    )
    
    ggplot() +
      geom_abline(slope = 1, intercept = 1, linetype = "dashed", color = "gray") +
      geom_line(
        data = train_df,
        aes(x = 1 - specificity, y = sensitivity, color = "Train"),
        linewidth = 1,
        alpha = 0.5
      ) +
      geom_line(
        data = test_df,
        aes(x = 1 - specificity, y = sensitivity, color = "Test"),
        linewidth = 1.2
      ) +
      scale_color_manual(
        values = c("Train" = "gray", "Test" = "blue"),
        name = NULL,
        labels = c(
          paste0("Train AUC: ", round(auc(res$train_roc), 3)),
          paste0("Test AUC: ", round(auc(res$test_roc), 3))
        )
      ) +
      labs(
        title = "ROC Curve",
        x = "False Positive Rate (1 - Specificity)",
        y = "True Positive Rate (Sensitivity)"
      ) +
      coord_equal() +
      theme_minimal(base_size = 14) +
      theme(legend.position = "bottom")
  })
  
  # Logistic Regression tabs
  output$coef_plot <- renderPlot({
    res <- model_results()
    if (res$model_type != "logistic") return(NULL)
    
    ms <- summary(res$model)$coefficients
    if (nrow(ms) < 2) return(NULL)
    
    coef_df <- data.frame(
      feature = rownames(ms)[-1],
      estimate = ms[-1, 1],
      std_error = ms[-1, 2],
      p_value = ms[-1, 4],
      stringsAsFactors = FALSE
    )
    coef_df$significant <- ifelse(coef_df$p_value < 0.05, "Yes", "No")
    coef_df <- coef_df[order(coef_df$estimate), ]
    coef_df$feature <- factor(coef_df$feature, levels = coef_df$feature)
    
    ggplot(coef_df, aes(x = estimate, y = feature, color = significant)) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
      geom_errorbarh(
        aes(
          xmin = estimate - 1.96 * std_error,
          xmax = estimate + 1.96 * std_error
        ),
        height = 0.25
      ) +
      geom_point(size = 3) +
      scale_color_manual(values = c("No" = "grey", "Yes" = "blue"), name = "p < 0.05") +
      labs(
        title = "Logistic Regression Coefficients with 95% CI",
        x = "Coefficient Estimate",
        y = ""
      ) +
      theme_minimal(base_size = 13) +
      theme(legend.position = "bottom")
  })
  
  output$summary_table <- renderTable({
    res <- model_results()
    if (res$model_type != "logistic") return(NULL)
    
    ms <- summary(res$model)$coefficients
    if (nrow(ms) < 2) return(NULL)
    
    coef_df <- data.frame(
      Feature = rownames(ms)[-1],
      Estimate = round(ms[-1, 1], 4),
      Std_Error = round(ms[-1, 2], 4),
      Z_value = round(ms[-1, 3], 3),
      P_value = ms[-1, 4],
      stringsAsFactors = FALSE
    )
    coef_df$Significance <- ifelse(
      coef_df$P_value < 0.001, "***",
      ifelse(
        coef_df$P_value < 0.01, "**",
        ifelse(
          coef_df$P_value < 0.05, "*",
          ifelse(coef_df$P_value < 0.1, ".", "")
        )
      )
    )
    coef_df$P_value <- formatC(coef_df$P_value, format = "e", digits = 3)
    coef_df <- coef_df[order(abs(coef_df$Estimate), decreasing = TRUE), ]
    coef_df
  }, striped = TRUE, bordered = TRUE, hover = TRUE)
  
  # Decision Tree tabs
  output$tree_plot <- renderPlot({
    res <- model_results()
    if (res$model_type != "tree") return(NULL)
    
    rpart.plot(
      res$model,
      type = 4,
      extra = 104,
      under = TRUE,
      roundint = FALSE,
      main = "Decision Tree",
      cex = 0.8,
      box.palette = "BuGn"
    )
  })
  
  output$varimp_plot <- renderPlot({
    res <- model_results()
    if (res$model_type != "tree") return(NULL)
    
    vi <- res$model$variable.importance
    if (is.null(vi) || length(vi) == 0) {
      plot.new()
      text(0.5, 0.5, "No variable importance available.\nTree may be too simple.", cex = 1.5)
      return(NULL)
    }
    
    vi_df <- data.frame(
      feature = names(vi),
      importance = as.numeric(vi),
      stringsAsFactors = FALSE
    )
    vi_df <- vi_df[order(vi_df$importance), ]
    vi_df$feature <- factor(vi_df$feature, levels = vi_df$feature)
    
    ggplot(vi_df, aes(x = importance, y = feature)) +
      geom_col(fill = "steelblue") +
      labs(
        title = "Variable Importance",
        x = "Importance",
        y = ""
      ) +
      theme_minimal(base_size = 13)
  })
  
  output$cp_table <- renderTable({
    res <- model_results()
    if (res$model_type != "tree") return(NULL)
    
    cp_df <- as.data.frame(res$model$cptable)
    cp_df$CP <- round(cp_df$CP, 6)
    cp_df$`rel error` <- round(cp_df$`rel error`, 4)
    cp_df$xerror <- round(cp_df$xerror, 4)
    cp_df$xstd <- round(cp_df$xstd, 4)
    cp_df
  }, striped = TRUE, bordered = TRUE, hover = TRUE)
}

# Run
shinyApp(ui = ui, server = server)
