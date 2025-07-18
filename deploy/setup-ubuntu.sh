#!/bin/bash
# Setup script for Ubuntu 22.04+ deployment of MES 0DTE Lotto-Grid Bot

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BOT_USER="lotto-grid"
BOT_HOME="/opt/lotto-grid-bot"
REPO_URL="https://github.com/user/lotto-grid-bot.git"  # Update with actual repo

echo -e "${GREEN}Starting MES 0DTE Lotto-Grid Bot deployment on Ubuntu...${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root (use sudo)${NC}"
    exit 1
fi

# Update system
echo -e "${YELLOW}Updating system packages...${NC}"
apt update && apt upgrade -y

# Install required packages
echo -e "${YELLOW}Installing required packages...${NC}"
apt install -y \
    curl \
    git \
    wget \
    unzip \
    htop \
    nano \
    fail2ban \
    ufw \
    logrotate \
    cron

# Install Docker
echo -e "${YELLOW}Installing Docker...${NC}"
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    systemctl enable docker
    systemctl start docker
else
    echo "Docker already installed"
fi

# Install Docker Compose
echo -e "${YELLOW}Installing Docker Compose...${NC}"
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
else
    echo "Docker Compose already installed"
fi

# Create bot user
echo -e "${YELLOW}Creating bot user...${NC}"
if ! id "$BOT_USER" &>/dev/null; then
    useradd -r -m -d "$BOT_HOME" -s /bin/bash "$BOT_USER"
    usermod -aG docker "$BOT_USER"
    echo "Created user $BOT_USER"
else
    echo "User $BOT_USER already exists"
fi

# Create directory structure
echo -e "${YELLOW}Setting up directory structure...${NC}"
mkdir -p "$BOT_HOME"/{data,logs,backups}
chown -R "$BOT_USER:$BOT_USER" "$BOT_HOME"

# Clone repository (if not exists)
if [ ! -d "$BOT_HOME/.git" ]; then
    echo -e "${YELLOW}Cloning repository...${NC}"
    sudo -u "$BOT_USER" git clone "$REPO_URL" "$BOT_HOME"
else
    echo "Repository already cloned"
fi

# Set up environment file
echo -e "${YELLOW}Setting up environment configuration...${NC}"
if [ ! -f "$BOT_HOME/.env" ]; then
    sudo -u "$BOT_USER" cp "$BOT_HOME/.env.example" "$BOT_HOME/.env"
    echo -e "${YELLOW}Please edit $BOT_HOME/.env with your IB credentials and settings${NC}"
fi

# Set up systemd service
echo -e "${YELLOW}Installing systemd service...${NC}"
cp "$BOT_HOME/deploy/systemd/lotto-grid-bot.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable lotto-grid-bot.service

# Set up log rotation
echo -e "${YELLOW}Configuring log rotation...${NC}"
cat > /etc/logrotate.d/lotto-grid-bot << EOF
$BOT_HOME/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    copytruncate
    su $BOT_USER $BOT_USER
}
EOF

# Set up firewall
echo -e "${YELLOW}Configuring firewall...${NC}"
ufw --force enable
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 8501/tcp comment 'Streamlit UI'
# ufw allow 5900/tcp comment 'VNC for IB Gateway (optional)'

# Set up fail2ban
echo -e "${YELLOW}Configuring fail2ban...${NC}"
systemctl enable fail2ban
systemctl start fail2ban

# Create backup script
echo -e "${YELLOW}Creating backup script...${NC}"
cat > "$BOT_HOME/scripts/backup.sh" << 'EOF'
#!/bin/bash
# Backup script for Lotto Grid Bot

BACKUP_DIR="/opt/lotto-grid-bot/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="lotto_grid_backup_${DATE}.tar.gz"

echo "Creating backup: $BACKUP_FILE"

# Stop bot
systemctl stop lotto-grid-bot

# Create backup
tar -czf "$BACKUP_DIR/$BACKUP_FILE" \
    -C /opt/lotto-grid-bot \
    data/ \
    logs/ \
    .env \
    --exclude='data/*.tmp' \
    --exclude='logs/*.tmp'

# Start bot
systemctl start lotto-grid-bot

# Keep only last 7 backups
find "$BACKUP_DIR" -name "lotto_grid_backup_*.tar.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_FILE"
EOF

chmod +x "$BOT_HOME/scripts/backup.sh"
chown "$BOT_USER:$BOT_USER" "$BOT_HOME/scripts/backup.sh"

# Set up daily backup cron job
echo -e "${YELLOW}Setting up daily backups...${NC}"
(crontab -u "$BOT_USER" -l 2>/dev/null; echo "0 2 * * * $BOT_HOME/scripts/backup.sh") | crontab -u "$BOT_USER" -

# Create update script
echo -e "${YELLOW}Creating update script...${NC}"
cat > "$BOT_HOME/scripts/update.sh" << 'EOF'
#!/bin/bash
# Update script for Lotto Grid Bot

cd /opt/lotto-grid-bot

echo "Stopping bot..."
systemctl stop lotto-grid-bot

echo "Backing up current version..."
./scripts/backup.sh

echo "Pulling latest code..."
git pull

echo "Rebuilding containers..."
docker-compose build --no-cache

echo "Starting bot..."
systemctl start lotto-grid-bot

echo "Update completed!"
EOF

chmod +x "$BOT_HOME/scripts/update.sh"
chown "$BOT_USER:$BOT_USER" "$BOT_HOME/scripts/update.sh"

# Set up monitoring script
echo -e "${YELLOW}Creating monitoring script...${NC}"
cat > "$BOT_HOME/scripts/monitor.sh" << 'EOF'
#!/bin/bash
# Monitoring script for Lotto Grid Bot

echo "=== Lotto Grid Bot Status ==="
echo "Service Status:"
systemctl status lotto-grid-bot --no-pager -l

echo -e "\nContainer Status:"
docker-compose ps

echo -e "\nResource Usage:"
docker stats --no-stream

echo -e "\nRecent Logs (last 20 lines):"
tail -20 /opt/lotto-grid-bot/logs/bot_run.log

echo -e "\nDisk Usage:"
df -h /opt/lotto-grid-bot
EOF

chmod +x "$BOT_HOME/scripts/monitor.sh"
chown "$BOT_USER:$BOT_USER" "$BOT_HOME/scripts/monitor.sh"

# Set proper ownership
chown -R "$BOT_USER:$BOT_USER" "$BOT_HOME"

# Build containers
echo -e "${YELLOW}Building Docker containers...${NC}"
cd "$BOT_HOME"
sudo -u "$BOT_USER" docker-compose build

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Edit the environment file: sudo nano $BOT_HOME/.env"
echo "2. Configure your IB credentials and trading parameters"
echo "3. Start the bot: sudo systemctl start lotto-grid-bot"
echo "4. Check status: sudo systemctl status lotto-grid-bot"
echo "5. Access UI at: http://your-server-ip:8501"
echo "6. Monitor logs: sudo journalctl -u lotto-grid-bot -f"
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo "- Start bot: sudo systemctl start lotto-grid-bot"
echo "- Stop bot: sudo systemctl stop lotto-grid-bot"
echo "- Restart bot: sudo systemctl restart lotto-grid-bot"
echo "- View logs: sudo journalctl -u lotto-grid-bot -f"
echo "- Update bot: sudo $BOT_HOME/scripts/update.sh"
echo "- Monitor status: sudo $BOT_HOME/scripts/monitor.sh"
echo "- Create backup: sudo $BOT_HOME/scripts/backup.sh"
echo ""
echo -e "${GREEN}Deployment guide completed!${NC}"
