library(readxl)
library(here)
library(tidyverse)
library(dplyr)
library(phyloseq)
library(writexl)
library(SummarizedExperiment)
library(lefser)








rm(list=ls())
source(here("SelectMicro_24new/Code/helpers.R"))

#######Analysis in WIN and SP independently (but compare soil and interface) ------------------


#######Read the data (Spring 16S)------------------
ARF_16S_sp_metadata <- read.csv("SelectMicro_24new/Analysis/ARF/data/ARF_16S_SP_metadata.csv",header = TRUE,row.names = 1)
ARF_16S_sp_ctb_family <- read.csv("SelectMicro_24new/Analysis/ARF/data/ARF_16S_ctb_SP_Family.csv",header = TRUE,row.names = 1)
# TSS count table (with cutoff 0.01)
ARF_16S_sp_ctb_family <- relative_abundance(ARF_16S_sp_ctb_family)
# append Depth, Phase and Donor to count tabe
colnames(ARF_16S_sp_metadata)
table(ARF_16S_sp_metadata$Phase,ARF_16S_sp_metadata$Depth)
# Check if row names match
if (all(rownames(ARF_16S_sp_ctb_family) == rownames(ARF_16S_sp_metadata))) {
  # Combine the DataFrames by columns
  ARF_16S_sp_ctb_family <- cbind(ARF_16S_sp_ctb_family, ARF_16S_sp_metadata[, c("Phase","Donor","Depth")])
} else {
  stop("Row names do not match.")
}
table(ARF_16S_sp_ctb_family$Phase)
#######remove initial and recovery stages------------------
ARF_16S_sp_ctb_family_filtered <- ARF_16S_sp_ctb_family[!(ARF_16S_sp_ctb_family$Phase %in% c("Initial", "RECOVERY","DECLINE")), ]
table(ARF_16S_sp_ctb_family_filtered$Phase,ARF_16S_sp_ctb_family_filtered$Depth)

ARF_16S_sp_metadata_filtered <- ARF_16S_sp_ctb_family_filtered[,c("Phase","Donor","Depth")]
ARF_16S_sp_ctb_family_filtered <- ARF_16S_sp_ctb_family_filtered[,!colnames(ARF_16S_sp_ctb_family_filtered) %in% c("Phase", "Donor", "Depth")]



# (1) Using classes only
res_class <- lefser(zeller14tn_ra,
                    classCol = "study_condition")
#> The outcome variable is specified as 'study_condition' and the reference category is 'CRC'.
#>  See `?factor` or `?relevel` to change the reference category.
# (2) Using classes and sub-classes
res_subclass <- lefser(zeller14tn_ra,
                       classCol = "study_condition",
                       subclassCol = "age_category")

## Create a SummarizedExperiment object
ARF_16S_sp <- SummarizedExperiment(assays = list(counts = t(ARF_16S_sp_ctb_family_filtered)), colData = ARF_16S_sp_metadata_filtered)
table(ARF_16S_sp_metadata_filtered$Phase)
table(ARF_16S_sp_metadata_filtered$Depth)
set.seed
classf <- as.factor(ARF_16S_sp_metadata_filtered$Phase)
lclassf <- levels(classf)
identical(length(lclassf), 2L)
res_16S_sp <- lefser(ARF_16S_sp, # relative abundance only with terminal nodes
              classCol = "Phase",
              subclassCol = "Depth")
head(res)








#######Read the data (Win 16S)------------------
ARF_16S_win_metadata <- read.csv("SelectMicro_24new/Analysis/ARF/data/ARF_16S_WIN_metadata.csv",header = TRUE,row.names = 1)
ARF_16S_win_ctb_family <- read.csv("SelectMicro_24new/Analysis/ARF/data/ARF_16S_ctb_WIN_Family.csv",header = TRUE,row.names = 1)
# TSS count table (with cutoff 0.01)
ARF_16S_win_ctb_family <- relative_abundance(ARF_16S_win_ctb_family)
# append Depth, Phase and Donor to count tabe
# Check if row names match
if (all(rownames(ARF_16S_win_ctb_family) == rownames(ARF_16S_win_metadata))) {
  # Combine the DataFrames by columns
  ARF_16S_win_ctb_family <- cbind(ARF_16S_win_ctb_family, ARF_16S_win_metadata[, c("Phase","Donor","Depth")])
} else {
  stop("Row names do not match.")
}
table(ARF_16S_win_ctb_family$Phase)

#######remove initial and recovery stages------------------
ARF_16S_win_ctb_family_filtered <- ARF_16S_win_ctb_family[!(ARF_16S_win_ctb_family$Phase %in% c("Initial", "RECOVERY")), ]




