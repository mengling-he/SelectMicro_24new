#############################################################################################################
#

rm(list=ls())

# Load library
library(phyloseq)
source(here("SelectMicro_24new/Code/helpers.R"))


# species
# Load  data of species
load("SelectMicro_24new/Analysis/Qitta/data/species/db11484.RData")
load("SelectMicro_24new/Analysis/Qitta/data/species/db1629.RData")
load("SelectMicro_24new/Analysis/Qitta/data/species/db2151.RData")

species_abundance_table = list(data.frame(db11484$abundance_table),
                               data.frame(db1629$abundance_table),data.frame(db2151$abundance_table))
species_abundance_table_combined <- bind_rows(species_abundance_table)
print(combined_df)


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
