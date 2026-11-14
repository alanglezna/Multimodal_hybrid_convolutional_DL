#!/bin/bash
# Híbrido final de un año. Se reinicia solo tras cada tarea (código 3) y se relanza si se cae (máximo 5 caídas).
# Uso:  nohup bash lanzar_final.sh Y21 > /dev/null 2>&1 &
ANIO=${1:-Y21}
mkdir -p logs
LOG="logs/log_final_${ANIO}.txt"
CAIDAS=0
while [ $CAIDAS -le 5 ]; do
  echo "=== Inicio final $ANIO: $(date) ===" >> "$LOG"
  ANIO_V7=$ANIO python3 -u correr_hibrido_final.py >> "$LOG" 2>&1
  CODIGO=$?
  if [ $CODIGO -eq 0 ]; then echo "=== final $ANIO terminada: $(date) ===" >> "$LOG"; exit 0; fi
  if [ $CODIGO -eq 3 ]; then sleep 5; continue; fi
  CAIDAS=$((CAIDAS + 1))
  echo "=== El proceso se cayó (código $CODIGO); reinicio $CAIDAS en 60 s: $(date) ===" >> "$LOG"
  sleep 60
done
echo "=== final $ANIO detenida tras 5 caídas: $(date) ===" >> "$LOG"
