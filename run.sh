python main.py \
--data_path ./data/ \
--epochs 8 \
--batch_size 1 \
--ngf 32 \
--ndf 32 \
--gan_loss lsgan \
--discriminator patch \
--scheduler none \
--use_l1_loss \
--use_perceptual_loss \
--use_style_loss \
--use_tv_loss \
--lambda_gan 0.01 \
--lambda_l1 1 \
--lambda_perceptual 0.1 \
--lambda_style 250 \
--lambda_tv 0.1 \
--width 256 \
--height 512 \
--n_downsamples_g 2 \
--lrG 0.0002 \
--lrD 0.00002 \
--beta1 0 \
--beta2 0.9 \
--norm instance \
--dilation 2 \
--kernel_size_g 4 \
--spectral_norm_d \
--attention_g \
--attention_d
