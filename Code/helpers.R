
# calculat the number of features(Otu,ITS) in each dataframe
get_n_features <- function(df) {
  
  n_features <- df %>%
    select(matches(c("Otu", "ITS"))) %>%
    summarise(features = ncol(.)) %>%
    pull()
  
  return(n_features)
}



# rename colnames of bact.n.class, bact.n.order, bact.n.phylum,
rename_tax_fun <- function(df, mapping_df, tax) {
  # Ensure the target_column exists in mapping_df
  if (!tax %in% colnames(mapping_df)) {
    stop("The specified target column does not exist in the mapping data frame.")
  }
  
  # Create the mapping vector using the specified column
  mapping_vector <- setNames(row.names(mapping_df),mapping_df[[tax]])
  # Make the new names unique by appending suffixes to duplicates
  unique_mapping_vector <- make.unique(names(mapping_vector))
  # Update the mapping vector with unique names
  names(mapping_vector) <- unique_mapping_vector
  
  # Rename the columns in df using dplyr
  df <- df %>% rename(any_of(mapping_vector))
  
  return(df)
}