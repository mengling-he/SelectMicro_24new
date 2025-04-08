rm(list=ls())

# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# 
# BiocManager::install("curatedMetagenomicData")

library(curatedMetagenomicData)



returnSamples
sample = sampleMetadata
study = unique(sample$study_name)
print(study)

# YachidaS_data_full <- curatedMetagenomicData(
#   pattern="YachidaS_2019",
#   dryrun = TRUE,
#   counts = FALSE,
#   rownames = "long"
# )
YachidaS_data <- curatedMetagenomicData(
  pattern="2021-10-14.YachidaS_2019.relative_abundance",
  dryrun = FALSE,
  counts = FALSE,
  rownames = "long"
)
YachidaS_data_data <- YachidaS_data$`2021-10-14.YachidaS_2019.relative_abundance`
count_data <- assay(YachidaS_data_data) 
count_data <- t(count_data)
dim(count_data)

metadata <- colData(YachidaS_data_data)  

write.csv(count_data, here("SelectMicro_24new/Analysis/Zeller/data/features_table.csv"),row.names = TRUE) 
write.csv(metadata, here("SelectMicro_24new/Analysis/Zeller/data/meta_data.csv"),row.names = TRUE) 

