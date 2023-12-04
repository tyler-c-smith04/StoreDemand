library(tidyverse)
library(vroom)
library(tidymodels)
library(forecast)
library(patchwork)
library(gridExtra)
library(ranger)
library(modeltime) #Extensions of tidymodels to time series
library(timetk) #Some nice time series functions
library(plotly)

train <- vroom("./train.csv")
test <- vroom("./test.csv")

# nStores <- max(train$store)
# nItems <- max(train$item)
# for(s in 1:nStores){
#   for(i in 1:nItems){
#     storeItemTrain <- train %>%
#       filter(store==s, item==i)
#     storeItemTest <- test %>%
#       filter(store==s, item==i)
# 
#     ## Fit storeItem models here
# 
#     ## Predict storeItem sales
# 
#     ## Save storeItem predictions
# 
#     if(s==1 & i==1){  
#       all_preds <- preds }else{
#         all_preds <- bind_rows(all_preds, preds)
#       }
# 
#   }
# }

plot1 <- train %>% 
  filter(item == 1, store == 1) %>%
  pull(sales) %>% 
  ggAcf(main = 'Item 1, Store 1')

plot2 <- train %>% 
  filter(item == 10, store == 2) %>%
  pull(sales) %>% 
  ggAcf(main = 'Item 10, Store 2')

plot3 <- train %>% 
  filter(item == 23, store == 4) %>%
  pull(sales) %>% 
  ggAcf(main = 'Item 23, Store 4')

plot4 <- train %>% 
  filter(item == 14, store == 9) %>%
  pull(sales) %>% 
  ggAcf(main = 'Item 14, Store 9')

combo <- (plot1 + plot2) / (plot3 + plot4)
combo

ggsave("acf_plots.png", combo)

storeItem <- train %>% 
  filter(store == 8, item == 30)

my_recipe <- recipe(sales ~., data = storeItem) %>%
  step_date(date, features="dow") %>%
  step_date(date, features="month") %>%
  step_date(date, features="year") %>%
  step_date(date, features="doy") %>%
  step_date(date, features="decimal") %>%
  step_date(date, features = "quarter") %>%
  step_range(date_doy, min = 0, max = pi) %>%
  step_mutate(sinDOY = sin(date_doy), cosDOY = cos(date_doy))

prepped_recipe <- prep(my_recipe)
baked <- bake(prepped_recipe, new_data = NULL)
baked

# Random Forest -----------------------------------------------------------
forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

# set workflow
forest_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(forest_mod)

## Grid of tuning values
tuning_grid <- grid_regular(mtry(range = c(1,10)),
                            min_n(),
                            levels = 5)

# split data into folds
folds <- vfold_cv(storeItem, v = 6, repeats = 1)

# run Cross validation
CV_results <- forest_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(smape))

# find best parameters
bestTune <- CV_results %>%
  select_best("smape")

bestTune

# collect metrics
collect_metrics(CV_results) %>%
  filter(mtry == 10, min_n == 30) %>%
  pull(mean)

# Exponential Smoothing ---------------------------------------------------
# Store 3, Item 17
storeItem <- train %>% filter(store==3, item==17)
cv_split <- time_series_split(storeItem, assess="3 months", cumulative = TRUE)
cv_split %>%
  tk_time_series_cv_plan() %>% #Put into a data frame
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)
  
es_model <- exp_smoothing() %>%
  set_engine("ets") %>%
  fit(sales~date, data=training(cv_split))
## Cross-validate to tune model
cv_results <- modeltime_calibrate(es_model,
                                  new_data = testing(cv_split))
## Visualize CV results
p1 <- cv_results %>%
  modeltime_forecast(
    new_data    = testing(cv_split),
    actual_data = storeItem
  )%>% plot_modeltime_forecast(.interactive=TRUE)

# Evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

# Refit to all data then forecast
es_fullfit <- cv_results %>% 
  modeltime_refit(data = storeItem)

es_preds <- es_fullfit %>% 
  modeltime_forecast(h= '3 months') %>% 
  rename(date = .index, sales = .value) %>% 
  select(date, sales) %>% 
  full_join(., y = test, by = 'date') %>% 
  select(id, sales)

p2 <- es_fullfit %>% 
  modeltime_forecast(h = '3 months', actual_data = storeItem) %>% 
  plot_modeltime_forecast(.interactive = FALSE)

p2
# Store 4, Item 4
storeItem2 <- train %>% filter(store==4, item==4)
cv_split <- time_series_split(storeItem2, assess="3 months", cumulative = TRUE)
cv_split %>%
  tk_time_series_cv_plan() %>% #Put into a data frame
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

es_model <- exp_smoothing() %>%
  set_engine("ets") %>%
  fit(sales~date, data=training(cv_split))
## Cross-validate to tune model
cv_results <- modeltime_calibrate(es_model,
                                  new_data = testing(cv_split))
## Visualize CV results
p3 <- cv_results %>%
  modeltime_forecast(
    new_data    = testing(cv_split),
    actual_data = storeItem2
  )%>% plot_modeltime_forecast(.interactive=TRUE)

# Evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

# Refit to all data then forecast
es_fullfit <- cv_results %>% 
  modeltime_refit(data = storeItem2)

es_preds <- es_fullfit %>% 
  modeltime_forecast(h= '3 months') %>% 
  rename(date = .index, sales = .value) %>% 
  select(date, sales) %>% 
  full_join(., y = test, by = 'date') %>% 
  select(id, sales)

p4 <- es_fullfit %>% 
  modeltime_forecast(h = '3 months', actual_data = storeItem2) %>% 
  plot_modeltime_forecast(.interactive = FALSE)

plotly::subplot(p1,p3,p2,p4, nrows = 2)

# Prophet -----------------------------------------------------------------

# Store 3, Item 17
storeItem <- train %>% filter(store==3, item==17)

cv_split <- time_series_split(storeItem, assess="3 months", cumulative = TRUE)
cv_split %>%
  tk_time_series_cv_plan() %>% #Put into a data frame
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

prophet_model <- prophet_reg() %>% 
  set_engine(engine = 'prophet') %>% 
  fit(sales ~ date, data = training(cv_split))

## Cross-validate to tune model
cv_results <- modeltime_calibrate(prophet_model,
                                  new_data = testing(cv_split))
## Visualize CV results
p1 <- cv_results %>%
  modeltime_forecast(
    new_data    = testing(cv_split),
    actual_data = storeItem
  )%>% plot_modeltime_forecast(.interactive=TRUE)

p1

# Evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

# Refit to all data then forecast
prophet_fullfit <- cv_results %>% 
  modeltime_refit(data = storeItem)

prophet_preds <- prophet_fullfit %>% 
  modeltime_forecast(h= '3 months') %>% 
  rename(date = .index, sales = .value) %>% 
  select(date, sales) %>% 
  full_join(., y = test, by = 'date') %>% 
  select(id, sales)

p2 <- prophet_fullfit %>% 
  modeltime_forecast(h = '3 months', actual_data = storeItem) %>% 
  plot_modeltime_forecast(.interactive = FALSE)

p2

# Store 4, Item 4
storeItem2 <- train %>% filter(store==4, item==4)

cv_split <- time_series_split(storeItem2, assess="3 months", cumulative = TRUE)
cv_split %>%
  tk_time_series_cv_plan() %>% #Put into a data frame
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

## Cross-validate to tune model
cv_results <- modeltime_calibrate(prophet_model,
                                  new_data = testing(cv_split))
## Visualize CV results
p3 <- cv_results %>%
  modeltime_forecast(
    new_data    = testing(cv_split),
    actual_data = storeItem2
  )%>% plot_modeltime_forecast(.interactive=TRUE)

p3

# Evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

# Refit to all data then forecast
prophet_fullfit <- cv_results %>% 
  modeltime_refit(data = storeItem2)

prophet_preds <- prophet_fullfit %>% 
  modeltime_forecast(h= '3 months') %>% 
  rename(date = .index, sales = .value) %>% 
  select(date, sales) %>% 
  full_join(., y = test, by = 'date') %>% 
  select(id, sales)

p4 <- prophet_fullfit %>% 
  modeltime_forecast(h = '3 months', actual_data = storeItem2) %>% 
  plot_modeltime_forecast(.interactive = FALSE)

p4

plotly::subplot(p1,p3,p2,p4, nrows = 2)


# Sarima -------------------------------------------------------------

## Read in the Data and filter to store/item
storeItemtrain2 <- train %>% filter(store==4, item==4)
storeItemtest2 <- test %>% filter(store==4, item==4)

cv_split <- time_series_split(storeItemtrain2, assess="3 months", cumulative = TRUE)
cv_split %>%
  tk_time_series_cv_plan() %>% #Put into a data frame
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

arima_recipe <- recipe(sales ~ ., storeItemtrain) %>% 
  step_date(date, features="month") %>% 
  step_date(date, features="year") %>% 
  step_date(date, features = "doy") %>%
  step_date(date, features="dow") %>% 
  step_range(date_doy, min = 0, max = pi) %>%
  step_mutate(sinDOY = sin(date_doy), cosDOY = cos(date_doy))

arima_model <- arima_reg(seasonal_period = 365,
                         non_seasonal_ar = 5, # default max p to tune
                         non_seasonal_ma = 5, # default max q to tune
                         seasonal_ar = 2, # default max P to tune
                         seasonal_ma = 2, # default max Q to tune
                         non_seasonal_differences = 2, # default max d to tune
                         seasonal_differences = 2) %>% # default max D to tune
  set_engine('auto_arima')

arima_wf <- workflow() %>% 
  add_recipe(arima_recipe) %>% 
  add_model(arima_model) %>% 
  fit(data = training(cv_split))

## Cross-validate to tune model
cv_results <- modeltime_calibrate(arima_wf,
                                  new_data = testing(cv_split))

## Visualize CV results
p3 <- cv_results %>%
  modeltime_forecast(
    new_data    = testing(cv_split),
    actual_data = storeItemtrain2
  )%>% plot_modeltime_forecast(.interactive=TRUE)

p3

fullfit <- cv_results %>% 
  modeltime_refit(data = storeItemtrain2)

## Predict for all observations in storeItemtest
p4 <- fullfit %>% 
  modeltime_forecast(
    new_data = storeItemtest2,
    actual_data = storeItemtrain2
  ) %>% 
  plot_modeltime_forecast(.interactive=TRUE)

## Store 3, Item 17
## Read in the Data and filter to store/item
storeItemtrain <- train %>% filter(store==3, item==17)
storeItemtest <- test %>% filter(store==3, item==17)

arima_recipe2 <- recipe(sales ~ ., storeItemtrain2) %>% 
  step_date(date, features="month") %>% 
  step_date(date, features="year") %>% 
  step_date(date, features = "doy") %>%
  step_date(date, features="dow") %>% 
  step_range(date_doy, min = 0, max = pi) %>%
  step_mutate(sinDOY = sin(date_doy), cosDOY = cos(date_doy))

cv_split <- time_series_split(storeItemtrain, assess="3 months", cumulative = TRUE)
cv_split %>%
  tk_time_series_cv_plan() %>% #Put into a data frame
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

arima_recipe <- recipe(sales ~date, data = storeItemtrain)

arima_model <- arima_reg(seasonal_period = 365,
                         non_seasonal_ar = 5, # default max p to tune
                         non_seasonal_ma = 5, # default max q to tune
                         seasonal_ar = 2, # default max P to tune
                         seasonal_ma = 2, # default max Q to tune
                         non_seasonal_differences = 2, # default max d to tune
                         seasonal_differences = 2) %>% # default max D to tune
  set_engine('auto_arima')

arima_wf <- workflow() %>% 
  add_recipe(arima_recipe2) %>% 
  add_model(arima_model) %>% 
  fit(data = training(cv_split))

## Cross-validate to tune model
cv_results <- modeltime_calibrate(arima_wf,
                                  new_data = testing(cv_split))

## Visualize CV results
p1 <- cv_results %>%
  modeltime_forecast(
    new_data    = testing(cv_split),
    actual_data = storeItemtrain
  )%>% plot_modeltime_forecast(.interactive=TRUE)

p1

fullfit <- cv_results %>% 
  modeltime_refit(data = storeItemtrain)

## Predict for all observations in storeItemtest
p2 <- fullfit %>% 
  modeltime_forecast(
    new_data = storeItemtest,
    actual_data = storeItemtrain
  ) %>% 
  plot_modeltime_forecast(.interactive=TRUE)

plotly::subplot(p1,p3,p2,p4, nrows = 2)


## Refit the data and forecast
preds <- cv_results %>%
  modeltime_refit(data = train) %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>% 
  full_join(., y=test, by="date") %>%
  select(id, sales)

# Prophet Submission ------------------------------------------------------
# Import Libraries

library(tidyverse)
library(tidymodels)
library(modeltime)
library(timetk)
library(vroom)

## Read in the data
item <- vroom::vroom("/kaggle/input/demand-forecasting-kernels-only/train.csv")
itemTest <- vroom::vroom("/kaggle/input/demand-forecasting-kernels-only/test.csv")

## How many?
n.stores <- max(item$store)
n.items <- max(item$item)

it <- 0
## Double Loop over all store-item combos
for(s in 1:n.stores){
  for(i in 1:n.items){
    
    ## Increment the progress bar
    it <- it + 1
    
    ## Subset the data
    train <- item %>%
      filter(store==s, item==i) %>%
      select(date, sales)
    test <- itemTest %>%
      filter(store==s, item==i) %>%
      select(id, date)
    
    ## CV Split
    cv_splits <- time_series_split(train, assess="3 months", cumulative = TRUE)
    
    ## Define and Tune Prophet Model
    prophet_model <- prophet_reg() %>%
      set_engine(engine = "prophet") %>%
      fit(sales ~ date, data = training(cv_splits))
    cv_results <- modeltime_calibrate(prophet_model, 
                                      new_data = testing(cv_splits))
    
    ## Refit the data and forecast
    preds <- cv_results %>%
      modeltime_refit(data = train) %>%
      modeltime_forecast(h = "3 months") %>%
      rename(date=.index, sales=.value) %>%
      select(date, sales) %>% 
      full_join(., y=test, by="date") %>%
      select(id, sales)
    
    ## If first store-item save, otherwise bind
    if(it==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds,
                             preds)
    }
    
    
  }
}

all_preds <- all_preds %>%
  arrange(id)

all_preds

vroom_write(x=all_preds, file="./submission.csv", delim=",")


# Boost Model -------------------------------------------------------------

## Define the workflow
item_recipe <- recipe(sales~., data=item) %>%
  step_date(date, features=c("dow", "month", "decimal", "doy", "year")) %>%
  step_range(date_doy, min=0, max=pi) %>%
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy)) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(sales)) %>%
  step_rm(date, item, store) %>%
  step_normalize(all_numeric_predictors())

boosted_model <- boost_tree(tree_depth=2, #Determined by random store-item combos
                            trees=1000,
                            learn_rate=0.01) %>%
  set_engine("lightgbm") %>%
  set_mode("regression")

boost_wf <- workflow() %>%
  add_recipe(item_recipe) %>%
  add_model(boosted_model)


# Exp Smoothing Preds -----------------------------------------------------

# Import Libraries
library(tidyverse)
library(tidymodels)
library(modeltime)
library(timetk)
library(vroom)
library(forecast)

# Read in the data
item <- vroom::vroom("./train.csv")
itemTest <- vroom::vroom("./test.csv")

# Convert date to Date type
item$date <- as.Date(item$date)
itemTest$date <- as.Date(itemTest$date)

# Determine the number of stores and items
n.stores <- max(item$store)
n.items <- max(item$item)

# Initialize an empty data frame to store predictions
all_preds <- data.frame()

# Loop over all store-item combos
for(s in 1:n.stores) {
  for(i in 1:n.items) {
    
    # Subset the data for the current store-item combination
    train <- item %>% filter(store == s, item == i)
    test <- itemTest %>% filter(store == s, item == i)
    
    # Check if there are any missing sales values
    if(any(is.na(train$sales))) {
      next # Skip to the next iteration if there are missing values
    }
    
    # Fit the Exponential Smoothing model
    ets_model <- ets(train$sales)
    
    # Forecast for the required horizon
    forecast_horizon <- nrow(test)
    forecasted_values <- forecast(ets_model, h = forecast_horizon)
    
    # Prepare the forecast for submission
    preds <- data.frame(id = test$id, sales = forecasted_values$mean)
    
    # Combine the predictions for each store-item combination
    all_preds <- bind_rows(all_preds, preds)
  }
}

all_preds

# Write the predictions to a CSV file
write.csv(all_preds, "submission.csv", row.names = FALSE)


# Sarima Preds ------------------------------------------------------------
# Import Libraries
library(tidyverse)
library(tidymodels)
library(modeltime)
library(timetk)
library(vroom)
library(forecast)

# Read in the data
item <- vroom::vroom("./train.csv")
itemTest <- vroom::vroom("./test.csv")

# Convert date to Date type
item$date <- as.Date(item$date)
itemTest$date <- as.Date(itemTest$date)

# Determine the number of stores and items
n.stores <- max(item$store)
n.items <- max(item$item)

# Initialize an empty data frame to store predictions
all_preds <- data.frame()

# Loop over all store-item combos
for(s in 1:n.stores) {
  for(i in 1:n.items) {
    
    # Subset the data for the current store-item combination
    train <- item %>% filter(store == s, item == i)
    test <- itemTest %>% filter(store == s, item == i)
    
    # Check if there are any missing sales values
    if(any(is.na(train$sales))) {
      next # Skip to the next iteration if there are missing values
    }
    
    # Fit the SARIMA model
    sarima_model <- auto.arima(train$sales, seasonal = TRUE)
    
    # Forecast for the required horizon
    forecast_horizon <- nrow(test)
    forecasted_values <- forecast(sarima_model, h = forecast_horizon)
    
    # Prepare the forecast for submission
    preds <- data.frame(id = test$id, sales = forecasted_values$mean)
    
    # Combine the predictions for each store-item combination
    all_preds <- bind_rows(all_preds, preds)
  }
}

all_preds

# Write the predictions to a CSV file
write.csv(all_preds, "sarima_predictions.csv", row.names = FALSE)


# Random Forest -----------------------------------------------------------
## Libraries
library(tidyverse)
library(tidymodels)
library(modeltime)
library(timetk)
library(vroom)
library(embed)
library(bonsai)
library(lightgbm)

## Read in the data
item <- vroom::vroom("./train.csv")
itemTest <- vroom::vroom("./test.csv")

n.stores <- max(item$store)
n.items <- max(item$item)

## Define the workflow
item_recipe <- recipe(sales~., data=item) %>%
  step_date(date, features=c("dow", "month", "decimal", "doy", "year")) %>%
  step_range(date_doy, min=0, max=pi) %>%
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy)) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(sales)) %>%
  step_rm(date, item, store) %>%
  step_normalize(all_numeric_predictors())

# Define Random Forest model
rf_model <- rand_forest() %>%
  set_mode("regression") %>%
  set_engine("randomForest")

rf_wf <- workflow() %>%
  add_recipe(item_recipe) %>%
  add_model(rf_model)

## Double Loop over all store-item combos
for(s in 1:n.stores){
  for(i in 1:n.items){
    
    ## Subset the data
    train <- item %>%
      filter(store==s, item==i)
    test <- itemTest %>%
      filter(store==s, item==i)
    
    ## Fit the data and forecast using Random Forest
    fitted_wf <- rf_wf %>%
      fit(data=train)
    preds <- predict(fitted_wf, new_data=test) %>%
      bind_cols(test) %>%
      rename(sales=.pred) %>%
      select(id, sales)
    
    ## Save the results
    if(s==1 && i==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds, preds)
    }
    
  }
}

vroom_write(x=all_preds, "submission.csv", delim=",")

