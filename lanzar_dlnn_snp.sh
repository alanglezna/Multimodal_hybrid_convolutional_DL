#!/bin/bash
# Lanza la DLNN con SNP en paralelo: 2 años x 2 espacios (compacto v6 y amplio v7) = 4 procesos en la misma GPU.
# Uso (en /workspace/trigo_v6, dentro del contenedor):
#     nohup bash lanzar_dlnn_snp.sh > /dev/null 2>&1 &
# Avance:   tail -n 3 logs/log_dlnn_snp_*.txt
# Para correr solo dos a la vez (si falta memoria):   PARALELO=2 nohup bash lanzar_dlnn_snp.sh > /dev/null 2>&1 &
# Si se interrumpe, se vuelve a lanzar: Optuna continúa donde iba y lo terminado se salta.
mkdir -p logs
PARALELO=${PARALELO:-4}

uno() {   # $1 = año, $2 = espacio
  local LOG="logs/log_dlnn_snp_${2}_${1}.txt" CAIDAS=0 CODIGO
  while [ $CAIDAS -le 5 ]; do
    echo "=== Inicio DLNN-SNP $2 $1: $(date) ===" >> "$LOG"
    ANIO=$1 ESPACIO=$2 python3 -u correr_dlnn_snp.py >> "$LOG" 2>&1
    CODIGO=$?
    if [ $CODIGO -eq 0 ]; then echo "=== DLNN-SNP $2 $1 terminada: $(date) ===" >> "$LOG"; return 0; fi
    if [ $CODIGO -eq 3 ]; then sleep 5; continue; fi
    CAIDAS=$((CAIDAS + 1))
    echo "=== El proceso se cayó (código $CODIGO); reinicio $CAIDAS en 60 s: $(date) ===" >> "$LOG"
    sleep 60
  done
  echo "=== DLNN-SNP $2 $1 detenida tras 5 caídas: $(date) ===" >> "$LOG"
}

N=0
for combo in "Y21 v6" "Y22 v6" "Y21 v7" "Y22 v7"; do
  set -- $combo
  if grep -q "terminada" "logs/log_dlnn_snp_${2}_${1}.txt" 2>/dev/null; then echo "Ya terminado: $combo"; continue; fi
  uno $1 $2 &
  N=$((N + 1))
  if [ $((N % PARALELO)) -eq 0 ]; then wait; fi
done
wait
echo "=== DLNN-SNP: TODO TERMINADO $(date) ==="
grep -h "r = " logs/log_dlnn_snp_*.txt
tar czf dlnn_snp_resultados.tar.gz Resultados_v6_Y2*/unimodales/DLNN-SNP_* Resultados_v7_Y2*/unimodales/DLNN-SNP_* \
    Opt_DLNN-SNP_v*_Y2*.db logs/log_dlnn_snp_*.txt 2>/dev/null
echo "Manda a Claude: dlnn_snp_resultados.tar.gz"
