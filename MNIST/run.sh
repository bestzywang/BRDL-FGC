#!/bin/bash
# CUDA_VISIBLE_DEVICES=4 PYTHONPATH="../../" bash .sh
# ps | grep -ie python | awk '{print $1}' | xargs kill -9 

# Verify the breakdown point
# The following codes are used to run exp_1.py and exp_2.py; Change niid and f to run them
function run_exp1 {
    COMMON_OPTIONS="--use-cuda --identifier all -n 15 -b 32 --momentum 0.99 --lr 0.05 --bucketing 0"
    for seed in  1 2 3
    do
        for atk in  "SF" "IPM" 
        do
            for f in   1 2 3 4 5 6
            do
                for agg in    "rfa" "tm" "cp" "cm" "krum" "avg" "nacp"
                do
                    for niid in  0.5
                    do
                        python exp_1.py $COMMON_OPTIONS --attack $atk --agg $agg -f $f --seed $seed --niid $niid &
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



# The following codes are used to run exp_3.py and exp_4.py; Change n and f to run them
function run_exp2 {
    COMMON_OPTIONS="--use-cuda --identifier all -n 15 -b 32 --niid 0.1 --lr 0.02 --bucketing 0 --noniid --clip-tau 0.215771"
    for seed in 1 2 3
    do
        for atk in "GA" "LF" "mimic" "ALIE" "SF" "IPM" 
        do
            for f in  1 2 3 4 5 6 7
            do
                for m in 0.99
                do
                    for agg in  "nacp"
                    do
                        python exp_3.py $COMMON_OPTIONS --attack $atk --agg $agg -f $f --seed $seed --momentum $m &
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

# The following codes are used for exp_5.py
function run_exp3 {
    COMMON_OPTIONS="--use-cuda --identifier all -n 15 -b 32 --lr 0.05 --bucketing 0 --niid 0.1 --noniid --clip-tau 0.215771"
    for seed in  3
    do
        for f in  6
        do
            for atk in "NA"
            do
                for m in 0.99
                do
                    for agg in  "rfa" "tm" "cp" "cm" "krum" "avg" "nacp"
                    do
                        python exp_5.py $COMMON_OPTIONS -f $f --agg $agg --attack $atk --seed $seed --momentum $m &
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

# The following codes are used to run exp_6.py and exp_7.py; Change niid and f to run them
function run_exp4 {
    COMMON_OPTIONS="--use-cuda --identifier all -n 15 -b 32 --momentum 0.99 --lr 0.05 --bucketing 0"
    for seed in  1 2 3
    do
        for atk in  "GA" "LF" "mimic" "ALIE" "SF" "IPM" 
        do
            for niid in  0.4 0.5 0.6 
            do
                for agg in    "nacp" "nacpt"
                do
                    for f in   3
                    do
                        python exp_6.py $COMMON_OPTIONS --attack $atk --agg $agg -f $f --seed $seed --niid $niid &
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


# The following codes are used for Table in the paper.
function run_exp5 {
    COMMON_OPTIONS="--use-cuda --identifier all -n 15 -b 32 --niid 0.5 --lr 0.05 --bucketing 0 --noniid --clip-tau 0.215771"
    for seed in 1 2 3
    do
        for atk in "GA" "LF" "mimic" "ALIE" "SF" "IPM" 
        do
            for f in 3
            do
                for m in 0.99
                do
                    for agg in   "rfa" "tm" "cp" "cm" "krum" "avg" "nacp"
                    do
                        python exp_t.py $COMMON_OPTIONS --attack $atk --agg $agg -f $f --seed $seed --momentum $m &
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
options=("exp1" "exp1_plot" "exp2" "exp2_plot" "exp3" "exp3_plot" "exp4" "exp4_plot" "exp5" "exp5_plot")
select opt in "${options[@]}"
do
    case $opt in
        "exp1")
            run_exp1
            ;;

        "exp1_plot")
            COMMON_OPTIONS="--use-cuda --identifier all -n 15 -f 1 --niid 0.5 --bucketing 0"
            python exp_1.py $COMMON_OPTIONS --plot
            ;;

        "exp2")
            run_exp2
            ;;

        "exp2_plot")
            COMMON_OPTIONS="--use-cuda --identifier all -n 15 -f 3 --niid 0.5 --clip-scaling linear"
            python exp_3.py $COMMON_OPTIONS --plot
            ;;

        "exp3")
            run_exp3
            ;;

        "exp3_plot")
            COMMON_OPTIONS="--use-cuda --identifier all -n 15 -f 0 "
            python exp_5.py $COMMON_OPTIONS --attack "IPM" --agg "cp" --plot
            ;;

        "exp4")
            run_exp4
            ;;

        "exp4_plot")
            COMMON_OPTIONS="--use-cuda --identifier all -n 15 -f 0 "
            python exp_6.py $COMMON_OPTIONS --attack "IPM" --agg "cp" --plot
            ;;

        "exp5")
            run_exp5
            ;;

        "exp5_plot")
            COMMON_OPTIONS="--use-cuda --identifier all -n 15 -f 0 "
            python exp_t.py $COMMON_OPTIONS --attack "IPM" --agg "cp" --plot
            ;;

        *) 
            echo "invalid option $REPLY"
            ;;
    esac
done


