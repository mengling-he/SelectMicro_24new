rm(list=ls())
# library(devtools)
# devtools::install_github("shbrief/lefserBenchmarking")


#install.packages("VennDiagram")
library(lefser)
library(microbiomeMarker)
library(lefserBenchmarking)
library(grid)
library(VennDiagram)
library(dplyr)
library(ggplot2)

data(zeller14)

# explore SummarizedExperiment object
count_data <- assay(zeller14)       # Retrieve count data
annotation_inf <- rowData(zeller14)     # Get gene annotations
metadata <- colData(zeller14)     # Get sample metadata




dim(count_data)
head(count_data)# each row is a feature (tax), each column is a sample
head(metadata)# each row is a sample, each column is a feature
dim(metadata)
head(annotation_inf)# here is the same with the row names of count_data


write.csv(count_data, here("SelectMicro_24new/Analysis/Zeller/data/features_table.csv"),row.names = TRUE) 
write.csv(metadata, here("SelectMicro_24new/Analysis/Zeller/data/meta_data.csv"),row.names = TRUE) 





table(zeller14$study_condition)

zeller14 <- zeller14[, zeller14$study_condition != "adenoma"]
# Run lefser

zeller14$study_condition <- factor(zeller14$study_condition,
                                   levels = c("CRC", "control")) # re-levels

res <- lefser(relativeAb(zeller14), 
              groupCol = "study_condition", 
              blockCol = "age_category")
lefserPlot(res)
#ggsave("Figures/Fig1A_CRC_lefser.png")

## Histogram
lefserPlotFeat(res, res$features[[1]])
lefserPlotFeat(res, res$features[[nrow(res)]])
#ggsave("Figures/Fig1C_CRC_histogram.png")



## Cladogram

tn <- get_terminal_nodes(rownames(zeller14))
zeller14_tn <- zeller14[tn,]
zeller14_tn_ra <- relativeAb(zeller14_tn)
zeller14_input <- rowNames2RowData(zeller14_tn_ra)

resAll <- lefserClades(zeller14_input, classCol = "study_condition")
lefserPlotClad(resAll, showNodeLabels = "o") 
ggsave("Figures/Fig1D_CRC_cladogram.png")

resAll2 <- lefserClades(zeller14_input, groupCol = "study_condition", blockCol = "age_category")
lefserPlotClad(resAll2, showTipLabels = TRUE)










# Run microbiomeMarker

ps <- formatInput(zeller14, "phyloseq")

set.seed(1982)
mm_lefse <- run_lefse(
  ps,
  wilcoxon_cutoff = 0.05,
  group = "study_condition",
  subgroup = "age_category",
  kw_cutoff = 0.05,
  multigrp_strat = TRUE,
  lda_cutoff = 2
)

plot_ef_bar(mm_lefse)
ggsave("Figures/SupFig3A_CRC_microbiomeMarker.png")
```


```{r echo=FALSE}
## Clean up the feature names from MM output
vectors <- marker_table(mm_lefse)$feature
vectors_updated <- lapply(vectors, function(x) {
  strsplit(x, "\\|") %>% unlist %>% tail(., 1)}) %>% unlist %>%
  stringr::str_replace(., "_p__.*$|_c__.*$|_o__.*$|_f__.*$|_g__.*$|_s__.*$", "")
```


# Comparisons
```{r}
lefser_output <- res %>% 
  mutate(app_name = 'lefser') %>% 
  arrange(scores) %>% 
  dplyr::rename(lefser_LDA = scores)
lefser_output$feature <- lapply(lefser_output$feature, function(x) {
  strsplit(x, "\\|") %>% unlist %>% tail(., 1)}) %>% unlist
```

```{r}
set1 <- lefse_docker$feature # LEfSe
set2 <- lefser_output$feature # lefser
set3 <- vectors_updated # biomarkers from microbiomeMarker

source("~/Projects/lefserBenchmarking/R/threeVennDiagram.R")
fit <- threeVennDiagram(set1, set2, set3)
plot(fit, quantities = TRUE)
```

## Annotations
```{r}
shared <- intersect(set1, set2)
shared_fullname <- lefser_output %>%
  filter(feature %in% shared) %>%
  pull(features)
```
