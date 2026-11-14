# =====================================================================
# R3 – Referencias adicionales para el artículo (los dos años), con los mismos pliegues y kernels de R1:
#   GBLUP-SNP (G de VanRaden), BLUP-DM6 (kernel lineal de DM6-AUC),
#   BLUP multimodal A + SNP + DM6 (A + G + lineal DM6-AUC), BLUP A + SNP y RKHS A + SNP.
# Así el artículo tiene BLUP y RKHS multimodales (como la versión enviada) y la comparación A + SNP
# contra el híbrido A + SNP con las mismas entradas.
# Uso (en /workspace/trigo_v6):  Rscript R3_referencias_extra.R
# Salida: Resultados_R_v6/pred_<año>_<modelo>.csv y tabla_referencia_extra.csv (+ por pliegue)
# =====================================================================
Sys.setenv(OPENBLAS_NUM_THREADS = 1)
suppressPackageStartupMessages({ library(BGLR); library(parallel) })
AÑOS <- c("Y21", "Y22"); NUCLEOS <- 12
NITER_UNI <- 6000; BURN_UNI <- 1500; NITER_MUL <- 30000; BURN_MUL <- 5000
D <- "Datos_entrenamiento/"; SALIDA <- "Resultados_R_v6"
dir.create(SALIDA, showWarnings = FALSE); dir.create("bglr_tmp", showWarnings = FALSE)

leer_mat <- function(ruta, gids) {
  d <- read.csv(ruta, check.names = FALSE); M <- as.matrix(d[, -1]); rownames(M) <- d$GID
  M[as.character(gids), , drop = FALSE]
}
DAT <- lapply(AÑOS, function(a) {
  dat <- readRDS(paste0(D, "kernels_", a, ".rds"))
  X <- leer_mat(paste0(D, "IMG_DM6AUC_", a, ".csv"), dat$gids)
  Xs <- scale(X); Xs[, apply(X, 2, sd) == 0] <- 0                 # mismas variables estandarizadas que el RKHS y la DLNN
  L <- tcrossprod(Xs) / ncol(Xs); dimnames(L) <- list(dat$gids, dat$gids)
  dat$KERNELS$LDM6AUC <- L; dat$EVD$LDM6AUC <- eigen(L, symmetric = TRUE)
  cat(a, ": kernel lineal DM6-AUC, diagonal media", round(mean(diag(L)), 3), "\n")
  dat
}); names(DAT) <- AÑOS

eta_de <- function(dat, ks) { ETA <- lapply(ks, function(k) { e <- dat$EVD[[k]]; keep <- e$values > 1e-7
  list(V = e$vectors[, keep], d = e$values[keep], model = "RKHS") }); names(ETA) <- ks; ETA }

MODELOS <- list(`BLUP_tri_A+SNP+DM6AUC` = c("A", "G", "LDM6AUC"),
                `BLUP_A+SNP` = c("A", "G"),
                `RKHS_A+SNP` = c("A", "K"),
                GBLUP_SNP = "G",
                BLUP_DM6AUC = "LDM6AUC")

tareas <- expand.grid(fold = 1:5, modelo = names(MODELOS), año = AÑOS, stringsAsFactors = FALSE)
tareas <- tareas[order(-sapply(tareas$modelo, function(m) length(MODELOS[[m]]))), ]
t0 <- Sys.time()
sal <- mclapply(seq_len(nrow(tareas)), function(j) {
  a <- tareas$año[j]; m <- tareas$modelo[j]; i <- tareas$fold[j]; dat <- DAT[[a]]; ks <- MODELOS[[m]]
  te <- which(dat$fold == i); yNA <- dat$y; yNA[te] <- NA
  multi <- length(ks) > 1; set.seed(345322 + i)
  fm <- BGLR(y = yNA, ETA = eta_de(dat, ks), nIter = if (multi) NITER_MUL else NITER_UNI,
             burnIn = if (multi) BURN_MUL else BURN_UNI, verbose = FALSE,
             saveAt = paste0("bglr_tmp/", a, "_", m, "_f", i, "_"))
  v <- sapply(ks, function(k) fm$ETA[[k]]$varU * mean(diag(dat$KERNELS[[k]])))
  list(año = a, modelo = m, fold = i, te = te, pred = fm$yHat[te], pct = 100 * v / sum(v))
}, mc.cores = NUCLEOS, mc.preschedule = FALSE)
if (any(sapply(sal, inherits, "try-error"))) stop("Falló alguna tarea")

filas <- list(); pliegue <- list()
for (a in AÑOS) for (m in names(MODELOS)) {
  dat <- DAT[[a]]; p <- numeric(length(dat$y)); pct <- NULL
  for (s in Filter(function(o) o$año == a && o$modelo == m, sal)) { p[s$te] <- s$pred; pct <- rbind(pct, s$pct) }
  write.csv(data.frame(GID = dat$gids, Fold_Prueba = dat$fold, trial.no = dat$ensayo, y_Real = dat$y, y_Pred = p),
            file.path(SALIDA, paste0("pred_", a, "_", m, ".csv")), row.names = FALSE)
  f <- function(fun) sapply(1:5, function(i) fun(dat$y[dat$fold == i], p[dat$fold == i]))
  rf <- f(cor); rm <- f(function(y, q) sqrt(mean((y - q)^2))); ma <- f(function(y, q) mean(abs(y - q)))
  pliegue[[paste(a, m)]] <- data.frame(año = a, modelo = m, pliegue = 1:5, r = rf, RMSE = rm, MAE = ma)
  filas[[paste(a, m)]] <- data.frame(año = a, modelo = m, r = mean(rf), sd_r = sd(rf), RMSE = mean(rm), MAE = mean(ma),
    var_pct = if (length(MODELOS[[m]]) > 1) paste(sprintf("%s %.1f", MODELOS[[m]], colMeans(pct)), collapse = " | ") else "")
}
T <- do.call(rbind, filas); T[, 3:6] <- round(T[, 3:6], 4); print(T, row.names = FALSE)
write.csv(T, file.path(SALIDA, "tabla_referencia_extra.csv"), row.names = FALSE)
write.csv(do.call(rbind, pliegue), file.path(SALIDA, "tabla_referencia_extra_por_pliegue.csv"), row.names = FALSE)
cat(sprintf("Listo en %.1f min\n", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
