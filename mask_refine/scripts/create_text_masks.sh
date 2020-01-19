python create_text_masks.py \
--data_path ../data/ \
--batch_size 20 \
--width 512 \
--height 1024 \
--dilation 2 \
--spectral_norm \
--ngf 32 \
--resume results/tversky_0.2/512x/checkpoints/checkpoint_epoch77.pth.tar \
