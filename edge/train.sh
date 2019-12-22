python main.py \
--data_path ../data/ \
--epochs 1 \
--batch_size 4 \
--gan_loss lsgan \
--scheduler none \
--use_fm_loss \
--lambda_gan 1 \
--lambda_fm 10 \
--width 256 \
--height 512 \
--lrG 0.0001 \
--lrD 0.00001 \
--beta1 0 \
--beta2 0.9 \
--dilation 2 \
--spectral_norm_d \
--spectral_norm_g \

