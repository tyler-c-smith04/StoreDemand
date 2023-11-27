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

# Sarima ------------------------------------------------------------------
# Store 3, Item 17
storeItem <- train %>% filter(store==3, item==17)

cv_split <- time_series_split(storeItem, assess="3 months", cumulative = TRUE)
cv_split %>%
  tk_time_series_cv_plan() %>% #Put into a data frame
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

arima_recipe <- recipe(sales ~date, data = storeItem)

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
arima_fullfit <- cv_results %>% 
  modeltime_refit(data = storeItem)

arima_preds <- arima_fullfit %>% 
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

cv_split <- time_series_split(storeItem, assess="3 months", cumulative = TRUE)
cv_split %>%
  tk_time_series_cv_plan() %>% #Put into a data frame
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)

arima_recipe2 <- recipe(sales ~date, data = storeItem2)

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
p3 <- cv_results %>%
  modeltime_forecast(
    new_data    = testing(cv_split),
    actual_data = storeItem
  )%>% plot_modeltime_forecast(.interactive=TRUE)

p3

# Evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = FALSE
  )

# Refit to all data then forecast
arima_fullfit <- cv_results %>% 
  modeltime_refit(data = storeItem2)

arima_preds <- arima_fullfit %>% 
  modeltime_forecast(h= '3 months') %>% 
  rename(date = .index, sales = .value) %>% 
  select(date, sales) %>% 
  full_join(., y = test, by = 'date') %>% 
  select(id, sales)

p4 <- es_fullfit %>% 
  modeltime_forecast(h = '3 months', actual_data = storeItem2) %>% 
  plot_modeltime_forecast(.interactive = FALSE)

p4

plotly::subplot(p1,p3,p2,p4, nrows = 2)
