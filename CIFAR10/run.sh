#!/bin/bash
# CUDA_VISIBLE_DEVICES=4 PYTHONPATH="../../" bash .sh
# ps | grep -ie python | awk '{print $1}' | xargs kill -9 

# The following codes are used for table iv in this paper
function run_exp1 {
    COMMON_OPTIONS="--use-cuda --identifier all -n 25 -b 32 --niid 0.3 --lr 0.25 --bucketing 0 --noniid --clip-tau 0.215771"
    for seed in  1 2 3
    do
        for atk in  "GA" "SF" "LF" "mimic" "IPM" "ALIE"
        do
            for f in 3 5
            do
                for m in 0.9
                do
                    for agg in  "rfa" "tm" "cp" "cm" "krum" "nacp"
                    do
                        python exp_1.py $COMMON_OPTIONS --attack $atk --agg $agg -f $f --seed $seed --momentum $m &
                        pids[$!]=$!
                    done
                done

                # wait for all pids
                for pid in ${pids[*]}; do
                    wait $pid
                done
                unset pids
            done
        done
    done
}

# The following codes are used for Fig.5 (b)
function run_exp2 {
    COMMON_OPTIONS="--use-cuda --identifier all -n 25 -b 32  --lr 0.5 --bucketing 0 --noniid --clip-tau 0.215771"
    for seed in 1 2 3
    do
        for nd in  0.3 0.5 0.7
        do
            for f in 6
            do
                for m in 0.9
                do
                    for agg in  "rfa" "tm" "cp" "cm" "krum" "nacp"
                    do
                        python exp_2.py $COMMON_OPTIONS --niid $nd --agg $agg -f $f --seed $seed --momentum $m &
                        pids[$!]=$!
                    done
                done

                # wait for all pids
                for pid in ${pids[*]}; do
                    wait $pid
                done
                unset pids
            done
        done
    done
}



PS3='Please enter your choice: '
options=("exp1" "exp1_plot" "exp2" "exp2_plot")
select opt in "${options[@]}"
do
    case $opt in
        
        "exp1")
            run_exp1
            ;;

        "exp1_plot")
            COMMON_OPTIONS="--use-cuda --identifier all -n 15 -f 1 --noniid --bucketing 0"
            python exp_1.py $COMMON_OPTIONS --plot
            ;;

        "exp2")
            run_exp2
            ;;

        "exp2_plot")
            COMMON_OPTIONS="--use-cuda --identifier all -n 15 -f 3 --noniid --clip-scaling linear"
            python exp_2.py $COMMON_OPTIONS --plot
            ;;

        *) 
            echo "invalid option $REPLY"
            ;;
    esac
done
