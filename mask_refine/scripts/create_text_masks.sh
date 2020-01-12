python create_text_masks.py \
--data_path ../data/ \
--batch_size 16 \
--width 512 \
--height 1024 \
--dilation 2 \
--spectral_norm \
--ngf 32 \
--resume results/tversky_0.2/checkpoints/checkpoint_epoch46.pth.tar \
