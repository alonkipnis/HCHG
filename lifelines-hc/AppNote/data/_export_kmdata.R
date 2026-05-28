
suppressPackageStartupMessages({
  if (!require("remotes", quietly = TRUE)) {
    install.packages("remotes", repos = "https://cloud.r-project.org", quiet = TRUE)
  }
  if (!require("kmdata", quietly = TRUE)) {
    cat("Installing kmdata from GitHub (raredd/kmdata)...\n")
    remotes::install_github("raredd/kmdata", quiet = TRUE)
  }
  library(kmdata)
})

trials  <- commandArgs(trailingOnly = TRUE)
out_dir <- trials[length(trials)]   # last arg is output dir
trials  <- trials[-length(trials)]  # remaining args are trial names

cat(sprintf("kmdata contains %d trials.\n", length(kmdata)))
cat(sprintf("Requested: %s\n", paste(trials, collapse = ", ")))

for (trial in trials) {
  if (!trial %in% names(kmdata)) {
    # Fuzzy match: try replacing _ with . or space
    alt <- names(kmdata)[grepl(gsub("_", ".", trial, fixed = TRUE),
                               names(kmdata), ignore.case = TRUE)]
    if (length(alt) == 0) {
      cat(sprintf("[SKIP] '%s' not found in kmdata. Available:\n", trial))
      cat(paste(sort(names(kmdata)), collapse = "\n"), "\n")
      next
    }
    trial <- alt[1]
    cat(sprintf("[INFO] Using '%s' as match.\n", trial))
  }
  df   <- kmdata[[trial]]
  fn   <- file.path(out_dir, paste0(trial, ".csv"))
  write.csv(df, fn, row.names = FALSE)
  cat(sprintf("[OK]   %s -> %s  (%d rows, cols: %s)\n",
              trial, fn, nrow(df), paste(colnames(df), collapse = ", ")))
}
