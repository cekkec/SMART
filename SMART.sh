conda activate xmem

python -m torch.distributed.run\
    --master_port 55900 \
    --nproc_per_node=2\
    train.py\
    --data_pick all \
    --set_loss SMART \
    --inter_rate 1 \
    --dice_rate 0.0001 \
    --uni_rate 0.0001 \
    --davis_root ../DAVIS \
    --yv_root ../YouTube \
    --tao_root ../TAO_VOS \
    --exp_id SMART_train \
    --load_network ./saves/XMem-s0.pth \
    --stage 3