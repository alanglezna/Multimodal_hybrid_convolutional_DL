#!/bin/bash
ANIO=$1
cd /workspace/trigo_v6
for p in 1 2; do
  if grep -q "terminada" "logs/log_v7_${ANIO}_parte${p}.txt" 2>/dev/null; then echo "Ya terminado: $ANIO $p"; continue; fi
  echo "=== Inicio $ANIO parte $p: $(date) ==="
  bash lanzar_v7.sh $ANIO $p
done
echo "=== $ANIO TERMINADO: $(date) ==="
