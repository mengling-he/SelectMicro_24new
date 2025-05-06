#############################################################################################################
#For this dataset there are 683 samples
rm(list=ls())

library(here)
library(dplyr)


# Load library
library(phyloseq)
source(here("SelectMicro_24new/Code/helpers.R"))

# genus
# Load  data of genus
load("SelectMicro_24new/Analysis/Qitta/data/genus/db1629.RData")
genus_abundance_table <- data.frame(db1629$abundance_table)
metadata_genus <- data.frame(db1629$metadata)
colnames(metadata_genus)
tax_db1629 = data.frame(db1629$taxonomy)
rownames(tax_db1629)
length(unique(tax_db1629$Rank5))#72
length(unique(tax_db1629$Rank6))#151

# preprocess of data ----------------
# replace missing Rank6 with previous taxonomy
value_counts_6 <- table(tax_db1629$Rank6)
repeated_tax <- names(value_counts_6[value_counts_6 > 1])

for (tax in repeated_tax){
  print(tax_db1629[tax_db1629$Rank6 == tax,])
  #tax_db1629[tax_db1629$Rank6 == tax,]$Rank6 = tax_db1629[tax_db1629$Rank6 == tax,]$Rank5
}
tax_db1629[tax_db1629$Rank6 == 'g__',]$Rank6 = tax_db1629[tax_db1629$Rank6 == 'g__',]$Rank5
tax_db1629[tax_db1629$Rank6 == 'o__',]$Rank6 = tax_db1629[tax_db1629$Rank6 == 'o__',]$Rank3
# Replace duplicated feature with "$_<rownumber>"
for (tax in repeated_tax){
  print(tax_db1629[tax_db1629$Rank6 == tax,])
  #tax_db1629[tax_db1629$Rank6 == tax,]$Rank6 = tax_db1629[tax_db1629$Rank6 == tax,]$Rank5
}
for (tax in repeated_tax){
  tax_db1629[tax_db1629$Rank6 == tax,]$Rank6 <- paste0(tax,'_', rownames(tax_db1629[tax_db1629$Rank6 == tax,]))
}

# rename the feature name of abundance matrix 
genus_abundance_table <- rename_tax_fun(genus_abundance_table,tax_db1629,'Rank6')
length(unique(colnames(genus_abundance_table)))# [1] 220


# delete the rows that are missing
missing_index_ibd <- which(is.na(metadata_genus$ibd))
genus_abundance_table_filter <- genus_abundance_table[-missing_index_ibd,]
metadata_genus_filter <- metadata_genus[-missing_index_ibd,]



write.csv(genus_abundance_table_filter, here("SelectMicro_24new/Analysis/Qitta/data/features_genus_db1629.csv"),row.names = TRUE) 
write.csv(metadata_genus_filter, here("SelectMicro_24new/Analysis/Qitta/data/meta_genus_db1629.csv"),row.names = TRUE) 




# Step 1: normalize and filter out based on cutoff
genus_abundance_norm <- relative_abundance(genus_abundance_table_filter)

combine_df <- as.data.frame(genus_abundance_norm)
combine_df$IBD <- metadata_genus_filter$ib
table(combine_df$IBD)







kruskal.test(colnames(combine_df)[10] ~ IBD,data=combine_df)
# Step 2: calculat the H statistics
kruskal_test_df <- function(df, response) {
  if (!is.factor(response)) {
    response <- as.factor(response)  # Ensure response is a factor
  }
  
  p_values <- sapply(df, function(column) {
    test_result <- kruskal.test(column ~ response)
    return(test_result$statistic)
  })
  
  return(p_values)
}

p_genus <- kruskal_test_df(genus_abundance_norm,metadata_genus_filter$ibd)
