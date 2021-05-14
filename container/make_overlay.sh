mkdir -p overlay/upper
mkdir -p overlay/work
dd if=/dev/zero of=$1 bs=1M count=100 && mkfs.ext3 -d overlay $1
