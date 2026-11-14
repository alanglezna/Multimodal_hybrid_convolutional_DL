#!/bin/bash
# Corre una parte del deep learning v6 para un año. El proceso de Python se reinicia solo después de cada tarea
# (código 3) para liberar memoria, y se relanza si se cae (máximo 5 caídas).
# Uso:  nohup bash lanzar_v6.sh Y21 1 > /dev/null 2>&1 &     (año: Y21 o Y22; parte: 1 = individuales, 2 = híbridos y Montesinos, 0 = todo)
# Avance: tail -f logs/log_v6_Y21_parte1.txt | grep --line-buffered -v ptx85
ANIO=${1:-Y21}
PARTE=${2:-0}
mkdir -p logs
LOG="logs/log_v6_${ANIO}_parte${PARTE}.txt"
CAIDAS=0
while [ $CAIDAS -le 5 ]; do
  echo "=== Inicio v6 $ANIO parte $PARTE: $(date) ===" >> "$LOG"
  ANIO_V6=$ANIO python3 -u -c "import correr_todo_v6 as ct; ct.correr(parte=$PARTE)" >> "$LOG" 2>&1
  CODIGO=$?
  if [ $CODIGO -eq 0 ]; then echo "=== v6 $ANIO parte $PARTE terminada: $(date) ===" >> "$LOG"; exit 0; fi
  if [ $CODIGO -eq 3 ]; then sleep 5; continue; fi
  CAIDAS=$((CAIDAS + 1))
  echo "=== El proceso se cayó (código $CODIGO); reinicio $CAIDAS en 60 s: $(date) ===" >> "$LOG"
  sleep 60
done
echo "=== v6 $ANIO parte $PARTE detenida tras 5 caídas: $(date) ===" >> "$LOG"
