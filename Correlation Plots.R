#libraries + read in file
library(tidyverse)
library(dplyr)
library(ggplot2)
library(tidymodels)
library(mlbench)
library(GGally)
library(readxl)
install.packages("corrplot")
library(corrplot)
install.packages("GGally")
library(GGally)

#load data
data_10k <- read_excel("Documents/Documents/Anything GMU/AIT 614/GCMF10k_6.xlsx")
View(data_10k)
head(data_10k)

######################################################################

dataf <- data.frame(data_10k[-c(2:12,17:21,26:38,40, 45:46, 49, 51)]) #remove variables that were unimportant to models
dataf #dropped expired_passport_marriage, citizenship_immigrant to make less messy 
View(dataf)

#now drop previously married, prev denied, foreign residence,
newframe <- dataf[-c(5:7)]
newframe
View(newframe)

#drop Sex for citizens and immigrants
framenew <- newframe[-c(12:13)]
View(framenew)


#ggpairs(dataf) #MESSY
#ggcorr(framenew, method = c("everything", "pearson")) #only gives numeric 

ggpairs(framenew, columns = 1:6, ggplot2::aes(colour=Fraud)) 
ggpairs(framenew, columns = 7:12, ggplot2::aes(colour=Fraud))




######################################################################
######################################################################
##########################numeric-only plots ###########################
######################################################################
######################################################################
#drop non-numeric columns to create corr matrix
df <- data_10k[-c(1:3,5:9,12,15:33,37:42,47:50)]
View(df)
new_df <- df[-c(13)]
View(new_df)

newnew_df <- new_df[-c(1:3,6:8)] #removed non-numeric and spouse stuff


#corr matrix
dta_mtrx <- cor(newnew_df)
dta_mtrx

#corr plot
corrplot(dta_mtrx, method = "number") 
#ggcorrplot(dta_mtrx)




