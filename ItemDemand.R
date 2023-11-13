library(tidyverse)
library(vroom)
library(tidymodels)
library(forecast)
library(patchwork)
library(gridExtra)

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

