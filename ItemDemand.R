library(tidyverse)
library(vroom)
library(tidymodels)
library(forecast)
library(patchwork)
library(gridExtra)
library(ranger)

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
  step_date(date, features = 'doy') %>% 
  step_range(date_doy, min=0, max=pi) %>% 
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy)) %>% 
  step_rm(date)

prepped_recipe <- prep(my_recipe)
baked <- bake(prepped_recipe, new_data = NULL)
baked

# Random Forest -----------------------------------------------------------
rand_forest_mod <- rand_forest(mtry = tune(),
                               min_n=tune(),
                               trees=1000) %>% # or 1000
  set_engine("ranger") %>%
  set_mode("regression")

rand_forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rand_forest_mod)

rand_forest_tuning_grid <- grid_regular(mtry(range = c(1, (ncol(storeItem)-1))),
                                        min_n(),
                                        levels = 5) ## L^2 total tuning possibilities

## Split data for CV
forest_folds <- vfold_cv(storeItem, v = 5, repeats = 1)

## Run the CV
CV_results <- rand_forest_wf %>%
  tune_grid(resamples = forest_folds,
            grid = rand_forest_tuning_grid,
            metrics = metric_set(smape)) # f_meas, sens, recall, spec, precision, accuracy

## Find Best Tuning Parameters
forest_bestTune <- CV_results %>%
  select_best("smape")

forest_bestTune

collect_metrics(CV_results) %>% 
  filter(mtry == 2, min_n == 40) %>% 
  pull(mean)
  


