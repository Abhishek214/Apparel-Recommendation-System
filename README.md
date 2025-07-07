TIMM_FUSE_DISABLED=1 HF_HUB_VERBOSITY=debug python train_efficientdet_local.py --config config.yaml --pretrained_model model.pth 2>&1 | grep -i "download\|fetch\|http"
