#############################################################################################################
#
rm(list=ls())

library(here)
library(dplyr)


# Load library
library(phyloseq)
source(here("SelectMicro_24new/Code/helpers.R"))


# Load the data from folder MLonMicrobiome/step2_count_preprocessing/"tax"/data_filtered99
# the features are already filtered considering 99% of zeros, i.e. the features with more than 99% of abundances equal to 0 have been removed from each dataset.


load("MLonMicrobiome/step2_count_preprocessing/species/data_filtered99/METADATA_table.RData")
load("MLonMicrobiome/step2_count_preprocessing/species/data_filtered99/OTU_table_sp_filt.RData")
load("MLonMicrobiome/step2_count_preprocessing/species/data_filtered99/TAX_table_sp_filt.RData")

for (i in 1:3){
  print(names(METADATA_table[i]))
  tax_table <- data.frame(TAX_table_sp_filt[[i]])
  if (!all(colnames(OTU_table_sp_filt[[i]]) %in% rownames(tax_table))) {
    warning(paste("Some samples in OTU table", i, "not found in taxtable Check colnames."))
  }
  tax_table$Rank <- paste(tax_table$Rank2,tax_table$Rank3,tax_table$Rank4,tax_table$Rank5, tax_table$Rank6, tax_table$Rank7, sep = "_")
  print(length(unique(tax_table$Rank))==nrow(tax_table))
  TAX_table_sp_filt[[i]] = tax_table
}

OTU_table_sp_filt[[2]]
METADATA_table[[2]]
names(METADATA_table)
names(OTU_table_sp_filt)
OTU_table_sp_filt[['db11484']]


for (i in names(METADATA_table)) { 
  i_metadata<- METADATA_table[[i]]
  print(colnames(i_metadata))
  }

for (i in names(METADATA_table)) { # For each dataset 
  # Select abundance table, taxonomy and metadata
  i_otu<- OTU_table_sp_filt[[i]]
  i_tax<- TAX_table_sp_filt[[i]]
  i_metadata<- METADATA_table[[i]]
  # Align metadata to OTU table by sample names (rows)
  if (!all(rownames(i_otu) %in% rownames(i_metadata))) {
    warning(paste("Some samples in OTU table", i, "not found in metadata. Check rownames."))
  }
  i_metadata<- i_metadata[rownames(i_otu),]# align metadata based on OTU row names
  # delete the rows that are missing
  missing_index_ibd <- which(is.na(i_metadata$ibd))
  abundance_table_filter <- i_otu[-missing_index_ibd,]
  metadata_filter <- i_metadata[-missing_index_ibd,]
  write.csv(abundance_table_filter, here(paste0("SelectMicro_24new/Analysis/Qitta/data/species/data_filtered99/features_", i, ".csv")),row.names = TRUE) 
  write.csv(i_tax, here(paste0("SelectMicro_24new/Analysis/Qitta/data/species/data_filtered99/tax_", i, ".csv")),row.names = TRUE) 
  write.csv(metadata_filter, here(paste0("SelectMicro_24new/Analysis/Qitta/data/species/data_filtered99/meta_", i, ".csv")),row.names = TRUE) 
}



# genus
load("MLonMicrobiome/step2_count_preprocessing/genus/data_filtered99/METADATA_table.RData")
load("MLonMicrobiome/step2_count_preprocessing/genus/data_filtered99/OTU_table_gen_filt.RData")
load("MLonMicrobiome/step2_count_preprocessing/genus/data_filtered99/TAX_table_gen_filt.RData")

for (i in 1:3){
  print(names(METADATA_table[i]))
  tax_table <- data.frame(TAX_table_gen_filt[[i]])
  if (!all(colnames(OTU_table_gen_filt[[i]]) %in% rownames(tax_table))) {
    warning(paste("Some samples in OTU table", i, "not found in taxtable Check colnames."))
  }
  tax_table$Rank <- paste(tax_table$Rank2,tax_table$Rank3,tax_table$Rank4,tax_table$Rank5, tax_table$Rank6, sep = "_")
  print(length(unique(tax_table$Rank))==nrow(tax_table))
  TAX_table_sp_filt[[i]] = tax_table
}



for (i in names(METADATA_table)) { 
  i_metadata<- METADATA_table[[i]]
  print(colnames(i_metadata))
}

for (i in names(METADATA_table)) { # For each dataset 
  # Select abundance table, taxonomy and metadata
  i_otu<- OTU_table_sp_filt[[i]]
  i_tax<- TAX_table_sp_filt[[i]]
  i_metadata<- METADATA_table[[i]]
  # Align metadata to OTU table by sample names (rows)
  if (!all(rownames(i_otu) %in% rownames(i_metadata))) {
    warning(paste("Some samples in OTU table", i, "not found in metadata. Check rownames."))
  }
  i_metadata<- i_metadata[rownames(i_otu),]# align metadata based on OTU row names
  # delete the rows that are missing
  missing_index_ibd <- which(is.na(i_metadata$ibd))
  abundance_table_filter <- i_otu[-missing_index_ibd,]
  metadata_filter <- i_metadata[-missing_index_ibd,]
  write.csv(abundance_table_filter, here(paste0("SelectMicro_24new/Analysis/Qitta/data/species/data_filtered99/features_", i, ".csv")),row.names = TRUE) 
  write.csv(i_tax, here(paste0("SelectMicro_24new/Analysis/Qitta/data/species/data_filtered99/tax_", i, ".csv")),row.names = TRUE) 
  write.csv(metadata_filter, here(paste0("SelectMicro_24new/Analysis/Qitta/data/species/data_filtered99/meta_", i, ".csv")),row.names = TRUE) 
}


















# species
# Load  data of species
load("SelectMicro_24new/Analysis/Qitta/data/species/db11484.RData")
load("SelectMicro_24new/Analysis/Qitta/data/species/db1629.RData")
load("SelectMicro_24new/Analysis/Qitta/data/species/db2151.RData")

species_abundance_table = list(data.frame(db11484$abundance_table),
                               data.frame(db1629$abundance_table),data.frame(db2151$abundance_table))
species_abundance_table_combined <- bind_rows(species_abundance_table)
print(species_abundance_table_combined)

metadata_species <- list(data.frame(db11484$metadata),
                         data.frame(db1629$metadata),data.frame(db2151$metadata))

colnames(metadata_species[[1]])
for (i in c(1,2,3)){
  print(table(metadata_species[[i]]$ibd))
}

metadata_species_combined <- bind_rows(metadata_species)
nonmissing_rows <- which(!is.na(metadata_species_combined$ibd))



tax_db11484 = data.frame(db11484$taxonomy)
tax_db1629 = data.frame(db1629$taxonomy)
identical(tax_db11484, tax_db1629)# same for 3 datasets
tax_db_species = tax_db11484
# Replace "S_" with "S_<rownumber>"
tax_db_species$Rank7[tax_db_species$Rank7 == "s__"] <- paste0("s__", rownames(tax_db_species)[tax_db_species$Rank7 == "s__"])


# rename each feature with species names
species_abundance_table_combined <- rename_tax_fun(species_abundance_table_combined,tax_db_species,'Rank7')
colnames(species_abundance_table_combined)
table(colnames(species_abundance_table_combined))






# genus
# Load  data of species
load("SelectMicro_24new/Analysis/Qitta/data/genus/db11484.RData")
load("SelectMicro_24new/Analysis/Qitta/data/genus/db1629.RData")
load("SelectMicro_24new/Analysis/Qitta/data/genus/db2151.RData")

genus_abundance_table = list(data.frame(db11484$abundance_table),
                               data.frame(db1629$abundance_table),data.frame(db2151$abundance_table))
genus_abundance_table_combined <- bind_rows(genus_abundance_table)

tax_db11484 = data.frame(db11484$taxonomy)
tax_db1629 = data.frame(db1629$taxonomy)
identical(tax_db11484, tax_db1629)# same for 3 datasets
tax_db_genus = tax_db11484
# Replace "S_" with "S_<rownumber>"
table(tax_db_genus$Rank6)
tax_db_genus$Rank6[tax_db_genus$Rank6 == "g__"] <- paste0("g__", rownames(tax_db_genus)[tax_db_genus$Rank6 == "g__"])


# rename each feature with species names
genus_abundance_table_combined <- rename_tax_fun(genus_abundance_table_combined,tax_db_genus,'Rank6')
colnames(genus_abundance_table_combined)

write.csv(species_abundance_table_combined[nonmissing_rows,], here("SelectMicro_24new/Analysis/Qitta/data/features_species_withname.csv"),row.names = TRUE) 

write.csv(genus_abundance_table_combined[nonmissing_rows,], here("SelectMicro_24new/Analysis/Qitta/data/features_genus_withname.csv"),row.names = TRUE) 

write.csv(metadata_species_combined[nonmissing_rows,], here("SelectMicro_24new/Analysis/Qitta/data/metadata.csv"),row.names = TRUE) 

