python evaluate.py \
--data_path ../data/ \
--batch_size 64 \
--gan_loss vanilla \
--width 256 \
--height 512 \
--dilation 2 \
--spectral_norm_d \
--spectral_norm_g \
--ngf 32 \
--ndf 32 \
--resume results/1/checkpoints/checkpoint_epoch24.pth.tar \