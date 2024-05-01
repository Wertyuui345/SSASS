# CAN ALSO THEORETICALLY USE $PMI_RANK, $SLURM_PROCID, $SLURM_NODEID TO REMOVE THE IF STATEMENTS, BUT RAN INTO ISSUES WHERE THOSE ENVIRONMENT VARIABLES WERE NOT POPULATING
MASTER_ADDR=$1 LOCAL_ADDR=$(hostname | cut -d '.' -f 1) #Master ADDR change based on argument passed in order.

echo first_local_addr_value @ $LOCAL_ADDR...
echo first master_addr_value @ $MASTER_ADDR...

if [ $MASTER_ADDR = $LOCAL_ADDR ]
then
    echo Launching master @ $MASTER_ADDR...
    RANK=0 python -m torch.distributed.launch --nproc_per_node=3 --nnodes=6 --master_addr=$MASTER_ADDR --node_rank=0 --master_port=3456 --use-env $SCRATCH/ls6/AudioSeperation/audiomodelmaskeddownmulti.py
elif [ $(scontrol show hostnames | sed -n '2p') = $LOCAL_ADDR ]
then
    echo Launching master @ $LOCAL_ADDR...
    RANK=3 python -m torch.distributed.launch --nproc_per_node=3 --nnodes=6 --master_addr=$MASTER_ADDR --node_rank=1 --master_port=3456 --use-env $SCRATCH/ls6/AudioSeperation/audiomodelmaskeddownmulti.py
elif [ $(scontrol show hostnames | sed -n '3p') = $LOCAL_ADDR ]
then
    echo Launching master @ $LOCAL_ADDR...
    RANK=6 python -m torch.distributed.launch --nproc_per_node=3 --nnodes=6 --master_addr=$MASTER_ADDR --node_rank=2 --master_port=3456 --use-env $SCRATCH/ls6/AudioSeperation/audiomodelmaskeddownmulti.py
elif [ $(scontrol show hostnames | sed -n '4p') = $LOCAL_ADDR ]
then
    echo Launching master @ $LOCAL_ADDR...
    RANK=9 python -m torch.distributed.launch --nproc_per_node=3 --nnodes=6 --master_addr=$MASTER_ADDR --node_rank=3 --master_port=3456 --use-env $SCRATCH/ls6/AudioSeperation/audiomodelmaskeddownmulti.py
elif [ $(scontrol show hostnames | sed -n '5p') = $LOCAL_ADDR ]
then
    echo Launching master @ $LOCAL_ADDR...
    RANK=12 python -m torch.distributed.launch --nproc_per_node=3 --nnodes=6 --master_addr=$MASTER_ADDR --node_rank=4 --master_port=3456 --use-env $SCRATCH/ls6/AudioSeperation/audiomodelmaskeddownmulti.py
elif [ $(scontrol show hostnames | sed -n '6p') = $LOCAL_ADDR ]
then
    echo Launching master @ $LOCAL_ADDR...
    RANK=15 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=6 --master_addr=$MASTER_ADDR --node_rank=5 --master_port=3456 --use-env $SCRATCH/ls6/AudioSeperation/audiomodelmaskeddownmulti.py
else
    echo No matching addr...
fi