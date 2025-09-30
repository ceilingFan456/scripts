# list all disks
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT

# -----------------------------
# Set your target partition here
DISK=/dev/sdc1        # e.g. /dev/sdc1
MOUNTPOINT=/mnt/data  # where you want it mounted
# -----------------------------

# If the disk is new and unformatted, create a partition (only once)
# Note: use ${DISK%?} to drop the "1" from /dev/sdc1 → /dev/sdc
sudo parted ${DISK%?} --script mklabel gpt mkpart primary ext4 0% 100%

# Format the partition (only once, if new)
sudo mkfs.ext4 -F $DISK

# Mount it
sudo mkdir -p $MOUNTPOINT
sudo mount $DISK $MOUNTPOINT

# Fix permissions
sudo chown -R $USER:$USER $MOUNTPOINT
sudo chmod 755 $MOUNTPOINT

# Get UUID
UUID=$(sudo blkid -s UUID -o value $DISK)
echo "Disk UUID = $UUID"

# Add UUID to /etc/fstab if not already present
LINE="UUID=$UUID  $MOUNTPOINT  ext4  defaults,nofail  0  2"
if ! grep -q "$UUID" /etc/fstab; then
  echo "$LINE" | sudo tee -a /etc/fstab
else
  echo "[*] UUID already exists in /etc/fstab"
fi

# Test fstab entry
sudo mount -a