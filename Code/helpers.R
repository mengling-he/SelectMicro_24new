
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


# calculate Relative abundance matrix (also in FS.py)
relative_abundance <- function(data, cutOff = 0.01) {
  # Check if the input data is a matrix or data.frame
  if (!is.matrix(data) && !is.data.frame(data)) {
    stop("Input 'data' must be a matrix or data frame.")
  }
  
  # Convert data to a matrix if it's a data.frame
  data <- as.matrix(data)
  
  # Calculate the total sum of each sample (row-wise sum)
  total_per_sample <- rowSums(data, na.rm = TRUE)
  
  # Check if all rows have zero total abundance
  if (all(total_per_sample == 0)) {
    stop("All rows have zero total abundance.")
  }
  
  # Normalize the data by dividing each value by the total sum for each row
  data_new <- sweep(data, 1, total_per_sample, FUN = "/")
  
  # Set values below the cutoff to 0
  data_new[data_new < cutOff] <- 0
  
  # Return the normalized matrix with NA replaced by 0
  return(data_new)
}