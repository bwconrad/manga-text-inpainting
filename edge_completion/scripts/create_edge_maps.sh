python create_edge_maps.py \
--data_path ../data/ \
--batch_size 20 \
--width 512 \
--height 1024 \
--dilation 2 \
--spectral_norm_g \
--ngf 32 \
--resume results/text_mask_tversky/512x/checkpoints/checkpoint_epoch44.pth.tar \
