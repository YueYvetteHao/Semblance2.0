#!/usr/bin/env Rscript
# Stage 2 of the pathway-activity build: per-cell ssGSEA over Reactome (C2:CP:REACTOME).
#
# Reads _work/expr_symbol.tsv (symbols x cells, from prep_expression.py), fetches the Reactome
# collection via msigdbr (same source as the Phase-1 DE/GSEA pipeline), runs GSVA::ssGSEA with
# cross-sample normalization OFF (strictly per-sample => leakage-clean), and writes a
# pathways x cells score matrix to _work/pathway_scores.tsv.
#
# ssGSEA input can be any monotone transform of TPM (it ranks within-sample), so log1p(TPM) is fine.
# Version-robust: handles both the msigdbr >=10 (collection/subcollection) and <10 (category/
# subcategory) APIs, and both the GSVA >=1.50 (ssgseaParam object) and older (method="ssgsea") APIs.
#
# Run on the box:  Rscript run_ssgsea.R
suppressMessages({
  library(data.table)
  library(msigdbr)
  library(GSVA)
})

here <- tryCatch(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 error = function(e) ".")
if (length(here) == 0 || here == "") here <- "."
work <- file.path(here, "_work")
in_tsv <- file.path(work, "expr_symbol.tsv")
out_tsv <- file.path(work, "pathway_scores.tsv")

cat("reading", in_tsv, "...\n")
dt <- fread(in_tsv, sep = "\t", header = TRUE)
symbols <- dt[[1]]
mat <- as.matrix(dt[, -1, with = FALSE])
rownames(mat) <- symbols
storage.mode(mat) <- "double"
cat(sprintf("  expression: %d symbols x %d cells\n", nrow(mat), ncol(mat)))

cat("fetching Reactome (C2:CP:REACTOME) via msigdbr ...\n")
msig <- tryCatch(
  msigdbr(species = "Homo sapiens", collection = "C2", subcollection = "CP:REACTOME"),
  error = function(e) msigdbr(species = "Homo sapiens", category = "C2", subcategory = "CP:REACTOME")
)
msig <- as.data.frame(msig)
gene_col <- if ("gene_symbol" %in% names(msig)) "gene_symbol" else "human_gene_symbol"
gsets <- split(msig[[gene_col]], msig$gs_name)
gsets <- lapply(gsets, unique)
cat(sprintf("  %d Reactome gene sets\n", length(gsets)))

ncores <- as.integer(Sys.getenv("SSGSEA_CORES", "24"))
bp <- tryCatch(BiocParallel::MulticoreParam(workers = ncores),
               error = function(e) BiocParallel::SerialParam())

cat(sprintf("running ssGSEA (normalize=FALSE, minSize=5, maxSize=500, workers=%d) ...\n", ncores))
scores <- tryCatch({
  # GSVA >= 1.50: parameter-object API
  par <- GSVA::ssgseaParam(exprData = mat, geneSets = gsets,
                           minSize = 5, maxSize = 500, normalize = FALSE)
  GSVA::gsva(par, BPPARAM = bp, verbose = FALSE)
}, error = function(e) {
  cat("  (falling back to legacy GSVA API:", conditionMessage(e), ")\n")
  GSVA::gsva(mat, gsets, method = "ssgsea", min.sz = 5, max.sz = 500,
             ssgsea.norm = FALSE, parallel.sz = ncores, verbose = FALSE)
})

cat(sprintf("  scores: %d pathways x %d cells\n", nrow(scores), ncol(scores)))
out <- data.table(pathway = rownames(scores), as.data.table(scores))
fwrite(out, out_tsv, sep = "\t")
cat("wrote", out_tsv, "\n")
