library(readxl)
library(here)
library(tidyverse)
library(dplyr)
library(phyloseq)
library(writexl)
source(here("SelectMicro_24new/Code/helpers.R"))


ARF_16S_metadata <- read.csv("SelectMicro_24new/Analysis/ARF/data/ARF_16S_metadata.csv",header = TRUE,row.names = 1)
ARF_16S_ctb_family <- read.csv("SelectMicro_24new/Analysis/ARF/data/ARF_16S_ctb_family.csv",header = TRUE,row.names = 1)


colnames(ARF_16S_metadata)

ARF_16S_metadata$Phase
table(ARF_16S_metadata$Phase)
table(ARF_16S_metadata$Donor)
table(ARF_16S_metadata$Depth)
