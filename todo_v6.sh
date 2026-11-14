#!/bin/bash
# Corre en orden: parte 1 Y21 -> parte 1 Y22 -> parte 2 Y21 -> parte 2 Y22.
# Si algún paso ya se está corriendo, espera a que termine; si ya terminó, lo salta.
cd /workspace/trigo_v6
while pgrep -f "lanzar_v6.sh" > /dev/null; do sleep 60; done      # espera si ya hay una corrida en marcha
for paso in "Y21 1" "Y22 1" "Y21 2" "Y22 2"; do
  set -- $paso
  if grep -q "terminada" "logs/log_v6_${1}_parte${2}.txt" 2>/dev/null; then
    echo "Ya terminado: $paso"; continue
  fi
  echo "=== Inicio $paso: $(date) ==="
  bash lanzar_v6.sh $1 $2
done
echo "=== TODO TERMINADO: $(date) ==="
