#*************************************************************
# IMPORT MOTHUR DATA AND SET UP PHYLOSEQ OBJECT FOR SP SERIES
#*************************************************************
# First, create the variables for the imported data (first 3 are ITS, second 3 are 16S)
sharedfile <- "NIJARFSP.trim.contigs.pcr.good.unique.precluster.pick.agc.shared"
taxfile <- "NIJARFSP.trim.contigs.pcr.good.unique.precluster.pick.agc.0.05.cons.taxonomy"
metadata <- read.csv(file="SP_ITS_metadata.csv")

sharedfile <- "NIJARFSP16S.shared"
taxfile <- "NIJARFSP16S.taxonomy"
metadata <- read.csv(file="SP_16S_metadata.csv")

# Now, import the mothur data
mothur_data <- import_mothur(mothur_shared_file = sharedfile, mothur_constaxonomy_file = taxfile)

# import the metadata file as a phyloseq object
metadata <- sample_data(metadata)
# In the metadata file set Sample_name as the row name
rownames(metadata) <- metadata$Sample_name

# Merge metadata file into phyloseq object created above
mothur_merged <-merge_phyloseq(mothur_data, metadata)
metadata
# Inspect column names of taxonomy file 
colnames(tax_table(mothur_merged))
# Current names are "Rank 1",...through "Rank 7"
# Rename them to something more accessible:
colnames(tax_table(mothur_merged))<- c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species") #ITS UNITE database
colnames(tax_table(mothur_merged))<- c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus") #16S Silva database

mothur_merged #This shows taxa and samples
sample_names(mothur_merged) # This shows what our samples are called and allows us to remove items as necessary (blanks, etc)
mothur_merged1a <- subset_samples(mothur_merged, Depth !="blank") #removes the sample blanks
mothur_merged1a # shows how many taxa and OTUs remain