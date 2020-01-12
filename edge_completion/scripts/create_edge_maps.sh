python create_edge_maps.py \
--data_path ../data/ \
--batch_size 16 \
--width 512 \
--height 1024 \
--dilation 2 \
--spectral_norm_g \
--ngf 32 \
--resume results/text_mask_tversky/checkpoints/checkpoint_epoch28.pth.tar \
