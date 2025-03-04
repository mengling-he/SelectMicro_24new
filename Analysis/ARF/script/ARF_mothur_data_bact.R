############################################################################
#########Filter samples and Merge the corresponding filtered 16S and ITS datasets
###########################################################################
library(readxl)
library(here)
library(tidyverse)
library(dplyr)
library(phyloseq)
library(writexl)
source(here("SelectMicro_24new/Code/helpers.R"))

############################################################################
#########some questions
# for Depth, what is blank and what is swab
###########################################################################

here()
#######Read the data (Spring 16S)------------------
# IMPORT MOTHUR DATA AND SET UP PHYLOSEQ OBJECT FOR SP SERIES
#*************************************************************
# First, create the variables for the imported data 

sharedfile_sp16 <- "SelectMicro_24new/Analysis/ARF/raw_data/NIJARFSP16S.shared"
taxfile_sp16 <- "SelectMicro_24new/Analysis/ARF/raw_data/NIJARFSP16S.taxonomy"
metadata_sp16 <- read.csv(here("SelectMicro_24new/Analysis/ARF/raw_data/SP_16S_metadata.csv"))

# Now, import the mothur data
mothur_data_sp16 <- import_mothur(mothur_shared_file = here(sharedfile_sp16), mothur_constaxonomy_file = here(taxfile_sp16))
mothur_data_sp16

# import the metadata file as a phyloseq object
metadata_sp16 <- sample_data(metadata_sp16)
# In the metadata file set Sample_name as the row name
rownames(metadata_sp16) <- metadata_sp16$Sample_name

# Merge metadata file into phyloseq object created above
mothur_merged_sp16 <-merge_phyloseq(mothur_data_sp16, metadata_sp16)
metadata_sp16
# Inspect column names of taxonomy file 
colnames(tax_table(mothur_merged_sp16))
# Current names are "Rank 1",...through "Rank 7"
# Rename them to something more accessible:
#colnames(tax_table(mothur_merged))<- c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species") #ITS UNITE database
colnames(tax_table(mothur_merged_sp16))<- c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus") #16S Silva database

mothur_merged_sp16 #This shows taxa and samples
sample_names(mothur_merged_sp16) # This shows what our samples are called and allows us to remove items as necessary (blanks, etc)
mothur_merged1b_sp16 <- subset_samples(mothur_merged_sp16, Depth !="blank" & Depth != "swab") #removes the sample blanks
mothur_merged1b_sp16 # shows how many taxa and OTUs remain
metadata_1b_sp16 <- sample_data(mothur_merged1b_sp16)
# Response variable: define the Phase based on values in Study_day and pass it to the metadata
metadata_1b_sp16$Phase <- ifelse(metadata_1b_sp16$Study_day %in% c(0), "Initial",
               ifelse(metadata_1b_sp16$Study_day %in% c(8, 12, 16), "BLOOM",
                      ifelse(metadata_1b_sp16$Study_day %in%c(27,43,58),'CLIMAX',
                             ifelse(metadata_1b_sp16$Study_day %in% c(72,86,103,117),'DECLINE','RECOVERY'))))

# make one feature to control for individual difference
# Depth can be used to control for depth difference
metadata_1b_sp16$Donor <- ifelse(metadata_1b_sp16$Sample %in% c('con15_mean','conint_mean'), "Control",
                                 ifelse(metadata_1b_sp16$Sample %in% c('gr_15_sp1', 'gr_int_sp1'), "Donor1",
                                        ifelse(metadata_1b_sp16$Sample %in%c('gr_15_sp2', 'gr_int_sp2'),'Donor2',
                                               ifelse(metadata_1b_sp16$Sample %in% c('gr_15_sp3', 'gr_int_sp3'),'Donor3','others'))))
sample_data(mothur_merged1b_sp16) <- metadata_1b_sp16
table(metadata_1b_sp16$Phase)
table(metadata_1b_sp16$Donor)
# use Depth(core/interface) and 
write.csv(data.frame(metadata_1b_sp16) , here("SelectMicro_24new/Analysis/ARF/data/ARF_16S_SP_metadata.csv"),row.names = TRUE)

# explore the object ------------
#*************************************************************
rank_names(mothur_merged_sp16)
sample_variables(mothur_merged1b_sp16)
table(metadata_1b_sp16$Sample)
table(metadata_1b_sp16$Study_day)
table(metadata_1b_sp16$Stage1)

table(metadata_1b_sp16$Stage)
table(metadata_1b_sp16$Stage.1)
# show the class on study day =8
plot_bar(subset_samples(mothur_merged1b_sp16, Study_day ==8), fill = "Phylum")


# OTU table of SP16S-------------
#*************************************************************
bact_phylo_sp <- mothur_merged1b_sp16
bact_phylo_sp_n=transform_sample_counts(bact_phylo_sp, function(x) {x/sum(x)*10000})#need to do total sum scaling: aka take relative abundance and then multiply by fixed library size of 10,000
bact_phylo_sp_n=prune_taxa(taxa_sums(bact_phylo_sp_n) > 10, bact_phylo_sp_n)#remove normalized OTUs with less than 10 reads across all samples 
#note went from 34,652 taxa to 6,183 taxa
sum(sample_sums(bact_phylo_sp_n)) # a total of 1559308 reads across all samples 


bact_OTU_sp_n=as.data.frame(otu_table(bact_phylo_sp_n))
if(taxa_are_rows(bact_phylo_sp_n)){bact_OTU_sp_n=t(bact_OTU_sp_n)}
bact_OTU_sp_n_df=as.data.frame(bact_OTU_sp_n)
bact.sp_OTU_tax_table = as.data.frame(tax_table(bact_phylo_sp_n))

dim(data.frame(metadata_1b_sp16))
dim(bact_OTU_sp_n_df)
dim(bact.sp_OTU_tax_table)

write.csv(bact_OTU_sp_n_df, here("SelectMicro_24new/Analysis/ARF/data/ARF_16S_ctb_SP_OTU.csv"),row.names = TRUE)
write.csv(bact.sp_OTU_tax_table, here("SelectMicro_24new/Analysis/ARF/data/ARF_16S_SP_taxtable.csv"),row.names = TRUE) 

# Class table of SP16S-------------
#*************************************************************
bact_phylo_class_sp = mothur_merged1b_sp16 %>% tax_glom(taxrank="Class")
bact_phylo_class_sp_n=transform_sample_counts(bact_phylo_class_sp, function(x) {x/sum(x)*10000})
bact_phylo_class_sp_n=prune_taxa(taxa_sums(bact_phylo_class_sp_n) > 10, bact_phylo_class_sp_n)
#note: went from 127 to 113 taxa

bact_class_sp_n=as(otu_table(bact_phylo_class_sp_n), "matrix")
if(taxa_are_rows(bact_phylo_class_sp_n)){bact_class_sp_n=t(bact_class_sp_n)}
bact_class_sp_n_df=as.data.frame(bact_class_sp_n)

bact.sp_c_tax_table = as.data.frame(tax_table(bact_phylo_class_sp_n))

dim(data.frame(metadata_1b_sp16))
dim(bact_class_sp_n_df)
dim(bact.sp_c_tax_table)




# Phylum table of SP16S-------------
#*************************************************************
bact_phylo_phylum_sp = mothur_merged1b_sp16 %>% tax_glom(taxrank="Phylum")
bact_phylo_phylum_sp_n=transform_sample_counts(bact_phylo_phylum_sp, function(x) {x/sum(x)*10000})
bact_phylo_phylum_sp_n=prune_taxa(taxa_sums(bact_phylo_phylum_sp_n) > 10, bact_phylo_phylum_sp_n)
#note: went from 41 to 39 taxa

bact_phylum_sp_n=as(otu_table(bact_phylo_phylum_sp_n), "matrix")
if(taxa_are_rows(bact_phylo_phylum_sp_n)){bact_phylum_sp_n=t(bact_phylum_sp_n)}
bact_phylum_sp_n_df=as.data.frame(bact_phylum_sp_n)

bact.sp_p_tax_table = as.data.frame(tax_table(bact_phylo_phylum_sp_n))

dim(data.frame(metadata_1b_sp16))
dim(bact_phylum_sp_n_df)
dim(bact.sp_p_tax_table)


# Family table of SP16S-------------
#*************************************************************
bact_phylo_family_sp = mothur_merged1b_sp16 %>% tax_glom(taxrank="Family")
bact_phylo_family_sp_n=transform_sample_counts(bact_phylo_family_sp, function(x) {x/sum(x)*10000})
bact_phylo_family_sp_n=prune_taxa(taxa_sums(bact_phylo_family_sp_n) > 10, bact_phylo_family_sp_n)
#note: went from 640 to 489 taxa

bact_family_sp_n=as(otu_table(bact_phylo_family_sp_n), "matrix")
if(taxa_are_rows(bact_phylo_family_sp_n)){bact_family_sp_n=t(bact_family_sp_n)}
bact_family_sp_n_df=as.data.frame(bact_family_sp_n)

bact.sp_f_tax_table = as.data.frame(tax_table(bact_phylo_family_sp_n))

dim(data.frame(metadata_1b_sp16))
dim(bact_family_sp_n_df)
dim(bact.sp_f_tax_table)



# Rename column names of count tables-------------
#*************************************************************
colnames(bact_class_sp_n_df)
colnames(bact_phylum_sp_n_df)
bact_class_sp_n_df <- rename_tax_fun(bact_class_sp_n_df,bact.sp_c_tax_table,'Class')
bact_phylum_sp_n_df <- rename_tax_fun(bact_phylum_sp_n_df,bact.sp_p_tax_table,'Phylum')
bact_family_sp_n_df <- rename_tax_fun(bact_family_sp_n_df,bact.sp_f_tax_table,'Family')
colnames(bact_OTU_sp_n_df)
colnames(bact_class_sp_n_df)
colnames(bact_phylum_sp_n_df)
colnames(bact_family_sp_n_df)

write.csv(bact_class_sp_n_df, here("SelectMicro_24new/Analysis/ARF/data/ARF_16S_ctb_SP_Class.csv"),row.names = TRUE)
write.csv(bact_phylum_sp_n_df, here("SelectMicro_24new/Analysis/ARF/data/ARF_16S_ctb_SP_Phylum.csv"),row.names = TRUE)
write.csv(bact_family_sp_n_df, here("SelectMicro_24new/Analysis/ARF/data/ARF_16S_ctb_SP_Family.csv"),row.names = TRUE)























#######Read the data (Win 16S)------------------
# IMPORT MOTHUR DATA AND SET UP PHYLOSEQ OBJECT FOR SP SERIES
#*************************************************************
# First, create the variables for the imported data 

sharedfile_win16 <- "SelectMicro_24new/Analysis/ARF/raw_data/NIJARFWIN16S.shared"
taxfile_win16 <- "SelectMicro_24new/Analysis/ARF/raw_data/NIJARFWIN16S.taxonomy"
metadata_win16 <- read.csv(here("SelectMicro_24new/Analysis/ARF/raw_data/WIN_16S_metadata.csv"))

# Now, import the mothur data
mothur_data_win16 <- import_mothur(mothur_shared_file = here(sharedfile_win16), mothur_constaxonomy_file = here(taxfile_win16))
mothur_data_win16

# import the metadata file as a phyloseq object
metadata_win16 <- sample_data(metadata_win16)
# In the metadata file set Sample_name as the row name
rownames(metadata_win16) <- metadata_win16$Sample_name

# Merge metadata file into phyloseq object created above
mothur_merged_win16 <-merge_phyloseq(mothur_data_win16, metadata_win16)
metadata_win16
# Inspect column names of taxonomy file 
colnames(tax_table(mothur_merged_win16))
# Current names are "Rank 1",...through "Rank 7"
# Rename them to something more accessible:
#colnames(tax_table(mothur_merged))<- c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species") #ITS UNITE database
colnames(tax_table(mothur_merged_win16))<- c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus") #16S Silva database

mothur_merged_win16 #This shows taxa and samples
sample_names(mothur_merged_win16) # This shows what our samples are called and allows us to remove items as necessary (blanks, etc)
mothur_merged1b_win16 <- subset_samples(mothur_merged_win16, Depth !="blank" & Depth != "swab") #removes the sample blanks

mothur_merged1b_win16 # shows how many taxa and OTUs remain
metadata_1b_win16 <- sample_data(mothur_merged1b_win16)

sample_variables(mothur_merged1b_win16)

# Response variable: define the Phase based on values in Study_day and pass it to the metadata
colnames(metadata_1b_win16)[colnames(metadata_1b_win16) == "Study.day"] <- "Study_day"
metadata_1b_win16$Phase <- ifelse(metadata_1b_win16$Study_day %in% c(0), "Initial",
                                 ifelse(metadata_1b_win16$Study_day %in% c(21, 38, 55), "BLOOM",
                                        ifelse(metadata_1b_win16$Study_day %in%c(75,94,110),'CLIMAX',
                                               ifelse(metadata_1b_win16$Study_day %in% c(126,140,158,172),'DECLINE','RECOVERY'))))

# make one feature to control for individual difference
# Depth can be used to control for depth difference
metadata_1b_win16$Donor <- ifelse(metadata_1b_win16$Sample %in% c('con15_mean','conint_mean'), "Control_win",
                                 ifelse(metadata_1b_win16$Sample %in% c('gr_15_win1', 'gr_int_win1'), "Donor1_win",
                                        ifelse(metadata_1b_win16$Sample %in%c('gr_15_win2', 'gr_int_win2'),'Donor2_win',
                                               ifelse(metadata_1b_win16$Sample %in% c('gr_15_win3', 'gr_int_win3'),'Donor3_win','others_win'))))

sample_data(mothur_merged1b_win16) <- metadata_1b_win16
table(metadata_1b_win16$Phase)
table(metadata_1b_win16$Donor)
table(metadata_1b_win16$Depth)


# OTU table of WIN16S-------------
#*************************************************************
bact_phylo_win <- mothur_merged1b_win16
bact_phylo_win_n=transform_sample_counts(bact_phylo_win, function(x) {x/sum(x)*10000})#need to do total sum scaling: aka take relative abundance and then multiply by fixed library size of 10,000
bact_phylo_win_n=prune_taxa(taxa_sums(bact_phylo_win_n) > 10, bact_phylo_win_n)#remove normalized OTUs with less than 10 reads across all samples 
#note went from 28,147 taxa to 5,385 taxa
sum(sample_sums(bact_phylo_win_n)) # a total of 1484632 reads across all samples 


bact_OTU_win_n=as.data.frame(otu_table(bact_phylo_win_n))
if(taxa_are_rows(bact_phylo_win)){bact_OTU_win_n=t(bact_OTU_win_n)}
bact_OTU_win_n_df=as.data.frame(bact_OTU_win_n)
bact.win_OTU_tax_table = as.data.frame(tax_table(bact_phylo_win_n))

dim(data.frame(metadata_1b_win16))
dim(bact_OTU_win_n_df)
dim(bact.win_OTU_tax_table)

write.csv(bact.win_OTU_tax_table, here("SelectMicro_24new/Analysis/ARF/data/ARF_16S_WIN_taxtable.csv"),row.names = TRUE) 

# Family table of WIN16S-------------
#*************************************************************
bact_phylo_family_win = mothur_merged1b_win16 %>% tax_glom(taxrank="Family")
bact_phylo_family_win_n=transform_sample_counts(bact_phylo_family_win, function(x) {x/sum(x)*10000})
bact_phylo_family_win_n=prune_taxa(taxa_sums(bact_phylo_family_win_n) > 10, bact_phylo_family_win_n)
#note: went from 620 to 459 taxa

bact_family_win_n=as(otu_table(bact_phylo_family_win_n), "matrix")
if(taxa_are_rows(bact_phylo_family_win_n)){bact_family_win_n=t(bact_family_win_n)}
bact_family_win_n_df=as.data.frame(bact_family_win_n)

bact.win_f_tax_table = as.data.frame(tax_table(bact_phylo_family_win_n))

dim(data.frame(metadata_1b_win16))
dim(bact_family_win_n_df)
dim(bact.win_f_tax_table)



# Class table of WIN16S-------------
#*************************************************************
bact_phylo_class_win = mothur_merged1b_win16 %>% tax_glom(taxrank="Class")
bact_phylo_class_win_n=transform_sample_counts(bact_phylo_class_win, function(x) {x/sum(x)*10000})
bact_phylo_class_win_n=prune_taxa(taxa_sums(bact_phylo_class_win_n) > 10, bact_phylo_class_win_n)
#note: went from 131 to 110 taxa

bact_class_win_n=as(otu_table(bact_phylo_class_win_n), "matrix")
if(taxa_are_rows(bact_phylo_class_win_n)){bact_class_win_n=t(bact_class_win_n)}
bact_class_win_n_df=as.data.frame(bact_class_win_n)

bact.win_c_tax_table = as.data.frame(tax_table(bact_phylo_class_win_n))

dim(data.frame(metadata_1b_win16))
dim(bact_class_win_n_df)
dim(bact.win_c_tax_table)

# Phylum table of WIN16S-------------
#*************************************************************
bact_phylo_phylum_win = mothur_merged1b_win16 %>% tax_glom(taxrank="Phylum")
bact_phylo_phylum_win_n=transform_sample_counts(bact_phylo_phylum_win, function(x) {x/sum(x)*10000})
bact_phylo_phylum_win_n=prune_taxa(taxa_sums(bact_phylo_phylum_win_n) > 10, bact_phylo_phylum_win_n)
#note: went from 42 to 38 taxa

bact_phylum_win_n=as(otu_table(bact_phylo_phylum_win_n), "matrix")
if(taxa_are_rows(bact_phylo_phylum_win_n)){bact_phylum_win_n=t(bact_phylum_win_n)}
bact_phylum_win_n_df=as.data.frame(bact_phylum_win_n)

bact.win_p_tax_table = as.data.frame(tax_table(bact_phylo_phylum_win_n))

dim(data.frame(metadata_1b_win16))
dim(bact_phylum_win_n_df)
dim(bact.win_p_tax_table)


# Rename column names of count tables-------------
#*************************************************************
colnames(bact_class_win_n_df)
colnames(bact_phylum_win_n_df)
colnames(bact_family_win_n_df)
bact_class_win_n_df <- rename_tax_fun(bact_class_win_n_df,bact.win_c_tax_table,'Class')
bact_phylum_win_n_df <- rename_tax_fun(bact_phylum_win_n_df,bact.win_p_tax_table,'Phylum')
bact_phylum_win_n_df <- rename_tax_fun(bact_family_win_n_df,bact.win_f_tax_table,'Family')

colnames(bact_class_win_n_df)
colnames(bact_phylum_win_n_df)
colnames(bact_family_win_n_df)







#######Combine SP and WIN samples for family------------------
# bact_family_win_n_df bact_family_sp_n_df
# Problem: Otu00001 in sp otu table is not the Otu00001 in winter otu table
# need to match them if want to combine the samples together
# right now can only develop models for SP and WIN separately
#*************************************************************
#* rename sample row names for SP and WIN samples

row.names(metadata_1b_win16)
row.names(bact_family_win_n_df)

row.names(metadata_1b_sp16)
row.names(bact_family_sp_n_df)

sp_16s_data_list <- list(metadata =metadata_1b_sp16, ctb_OTU=bact_OTU_sp_n_df,
                         ctb_family=bact_family_sp_n_df,ctb_class = bact_class_sp_n_df,
                         ctb_phylum = bact_phylum_sp_n_df)


win_16s_data_list <- list(metadata =metadata_1b_win16, ctb_OTU=bact_OTU_win_n_df,
                         ctb_family=bact_family_win_n_df,ctb_class = bact_class_win_n_df,
                         ctb_phylum = bact_phylum_win_n_df)

sp_16s_data_list <- lapply(sp_16s_data_list, function(df) {
  rownames(df) <- paste0("SP_", rownames(df))
  return(df)
})

win_16s_data_list <- lapply(win_16s_data_list, function(df) {
  rownames(df) <- paste0("WIN_", rownames(df))
  return(df)
})

metadata_16S_sp <- sp_16s_data_list[1]
ctb_OTU_sp <- sp_16s_data_list[2]
ctb_family_sp <- sp_16s_data_list[3]
ctb_class_sp <- sp_16s_data_list[4]
ctb_phylum_sp <- sp_16s_data_list[5]

metadata_16S_win <- win_16s_data_list[1]
ctb_OTU_win <- win_16s_data_list[2]
ctb_family_win <- win_16s_data_list[3]
ctb_class_win <- win_16s_data_list[4]
ctb_phylum_win <- win_16s_data_list[5]

write.csv(metadata_16S_sp , here("SelectMicro_24new/Analysis/ARF/data/ARF_16S_SP_metadata.csv"),row.names = TRUE)
write.csv(ctb_OTU_sp, here("SelectMicro_24new/Analysis/ARF/data/ARF_16S_ctb_SP_OTU.csv"),row.names = TRUE)
write.csv(ctb_family_sp, here("SelectMicro_24new/Analysis/ARF/data/ARF_16S_ctb_SP_Family.csv"),row.names = TRUE)
write.csv(ctb_class_sp, here("SelectMicro_24new/Analysis/ARF/data/ARF_16S_ctb_SP_Class.csv"),row.names = TRUE)
write.csv(ctb_phylum_sp, here("SelectMicro_24new/Analysis/ARF/data/ARF_16S_ctb_SP_Phylum.csv"),row.names = TRUE)


write.csv(metadata_16S_win , here("SelectMicro_24new/Analysis/ARF/data/ARF_16S_WIN_metadata.csv"),row.names = TRUE)
write.csv(ctb_OTU_win, here("SelectMicro_24new/Analysis/ARF/data/ARF_16S_ctb_WIN_OTU.csv"),row.names = TRUE)
write.csv(ctb_family_win, here("SelectMicro_24new/Analysis/ARF/data/ARF_16S_ctb_WIN_Family.csv"),row.names = TRUE)
write.csv(ctb_class_win, here("SelectMicro_24new/Analysis/ARF/data/ARF_16S_ctb_WIN_Class.csv"),row.names = TRUE)
write.csv(ctb_phylum_win, here("SelectMicro_24new/Analysis/ARF/data/ARF_16S_ctb_WIN_Phylum.csv"),row.names = TRUE)

  
