mkdir -p overlay/upper
mkdir -p overlay/work
dd if=/dev/zero of=overlay.img bs=1M count=100 && mkfs.ext3 -d overlay overlay.img
