#!/usr/bin/env Rscript
# Atlas step 2 (R): pre-ranked GSEA over Reactome for each Spearman-ranked (disease, drug) list.
#
# Consumes _work/ranks/*.rnk (symbol<TAB>rho, from prep_gsea_ranks.py; rho = Spearman(gene,-sens_z),
# positive = sensitivity-associated) and runs fgsea (fast pre-ranked GSEA) with the same Reactome
# gene sets the Phase-1 pipeline used. Harmonized, sign-corrected, GDSC∪CTRP successor to Phase-1
# (which was GDSC-only, all-cells, DESeq2 median-split). INTERPRETATION ONLY — never a model feature.
#
# Uses fgsea (not clusterProfiler) — same pre-ranked GSEA, but no enrichplot/ggtree/cairo stack, so
# it installs cleanly on arm64. Writes out/gsea_signatures/<disease>/<drug_uid>_GSEA_REACTOME.csv
# (Phase-1-like schema) + out/gsea_signatures/MASTER_SIGNATURES.csv. Parallel across pairs.
#
# Run on the box:  Rscript run_gsea_signatures.R
suppressMessages({
  library(data.table)
  library(msigdbr)
  library(fgsea)
})

here <- tryCatch(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
                 error = function(e) ".")
if (length(here) == 0 || here == "") here <- "."
work <- file.path(here, "_work")
ranks_dir <- file.path(work, "ranks")
out_dir <- file.path(here, "out", "gsea_signatures")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

manifest <- fread(file.path(work, "ranks_manifest.csv"))
cat(sprintf("fgsea over %d ranked lists\n", nrow(manifest)))

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

ncores <- as.integer(Sys.getenv("GSEA_CORES", "24"))

one <- function(i) {
  row <- manifest[i]
  rnk <- fread(file.path(ranks_dir, row$rnk), header = FALSE, col.names = c("symbol", "rho"))
  rnk <- rnk[is.finite(rho)]
  rnk <- rnk[!duplicated(symbol)]
  gl <- rnk$rho
  names(gl) <- rnk$symbol
  gl <- sort(gl, decreasing = TRUE)
  res <- tryCatch(
    fgsea(pathways = gsets, stats = gl, minSize = 5, maxSize = 500, eps = 0),
    error = function(e) NULL)
  if (is.null(res) || nrow(res) == 0) return(NULL)
  res <- as.data.table(res)
  res[, leadingEdge := vapply(leadingEdge, function(x) paste(x, collapse = "/"), character(1))]
  # Phase-1-like schema
  df <- res[order(-NES), .(ID = pathway, Description = pathway, setSize = size,
                           enrichmentScore = ES, NES = NES, pvalue = pval, p.adjust = padj,
                           core_enrichment = leadingEdge)]
  ddir <- file.path(out_dir, gsub("/", "-", row$disease))
  dir.create(ddir, showWarnings = FALSE, recursive = TRUE)
  fwrite(df, file.path(ddir, sprintf("%s_GSEA_REACTOME.csv", row$drug_uid)))
  top <- df[1]
  data.table(disease = row$disease, drug_uid = row$drug_uid, drug_name = row$drug_name,
             n_lines = row$n_lines, n_sets = nrow(df), n_sig05 = sum(df$p.adjust < 0.05, na.rm = TRUE),
             top_pathway = top$ID, top_NES = top$NES, top_padj = top$p.adjust)
}

res_list <- parallel::mclapply(seq_len(nrow(manifest)), one, mc.cores = ncores)
master <- rbindlist(Filter(Negate(is.null), res_list), fill = TRUE)
fwrite(master, file.path(out_dir, "MASTER_SIGNATURES.csv"))
cat(sprintf("wrote %d signature tables + MASTER_SIGNATURES.csv to %s\n", nrow(master), out_dir))
