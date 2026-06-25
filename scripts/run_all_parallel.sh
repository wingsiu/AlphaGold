#!/bin/bash
rm -f parallel_sweep_results.txt
echo "Horizon,TP,SL,Trades,WinRate,NetPnL" > parallel_sweep_results.txt

horizons=(15 30 45)
targets=(15 30 45)
stops=(10 15 25)

count=0
for h in "${horizons[@]}"; do
    for t in "${targets[@]}"; do
        for s in "${stops[@]}"; do
            if (( s >= t )); then
                continue
            fi
            count=$((count+1))
            ./run_parallel_sweep.sh $count $h $t $s >> parallel_sweep_results.txt &
            
            # Limit to 8 concurrent jobs to avoid crashing
            if (( count % 8 == 0 )); then
                wait
            fi
        done
    done
done
wait
echo "Sweep complete!"
