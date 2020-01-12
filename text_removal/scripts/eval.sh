python eval.py \
--data_path ../data/ \
--batch_size 16 \
--ngf 32 \
--ndf 32 \
--generator unet \
--discriminator patch \
--width 256 \
--height 512 \
--n_downsamples_g 2 \
--norm instance \
--dilation 2 \
--kernel_size_g 4 \
--spectral_norm_d \
--edges \
--eval_dataset val \
--eval_save \
--resume results/unet_refine_mask_edgeconnect_edge_patch_tr/checkpoints/checkpoint_epoch34.pth.tar
