#!/bin/bash

# docker-scripts.sh - Docker management utilities

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
ENV_FILE="$SCRIPT_DIR/.env"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker is not running or not accessible"
        exit 1
    fi
    
    if ! command -v docker compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available"
        exit 1
    fi
}

# Function to validate environment file
validate_env() {
    if [[ ! -f "$ENV_FILE" ]]; then
        log_warn "Environment file not found: $ENV_FILE"
        log_info "Creating default environment file..."
        
        cat > "$ENV_FILE" << 'EOF'
# Docker Environment Configuration
MEMORY_LIMIT_GB=8
CPU_LIMIT=4
MAX_WORKERS=2
N_THREADS=4
DATA_DIR=./data
DB_HOST=postgres
DB_PORT=5432
DB_NAME=dataindex
DB_USER=postgres
DB_PASSWORD=change_me_please
SSH_TUNNEL=false
SSH_KEY_DIR=~/.ssh
LOG_LEVEL=INFO
JSON_LOGGING=false
PYTHONPATH=/app
PYTHONUNBUFFERED=1
EOF
        log_warn "Please edit $ENV_FILE with your configuration before proceeding"
        return 1
    fi
    
    # Check for required variables
    source "$ENV_FILE"
    local required_vars=("DATA_DIR" "DB_PASSWORD")
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        log_error "Please update $ENV_FILE"
        return 1
    fi
    
    # Validate data directory
    if [[ ! -d "$DATA_DIR" ]]; then
        log_error "Data directory not found: $DATA_DIR"
        log_error "Please create the directory or update DATA_DIR in $ENV_FILE"
        return 1
    fi
    
    return 0
}

# Function to build the Docker image
build_image() {
    log_info "Building Docker image..."
    if docker build --no-cache -f offline/Dockerfile -t data-lake-indexer "."; then
        log_success "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to start services
start_services() {
    local services=("${@:-}")
    log_info "Starting services..."
    
    if [[ ${#services[@]} -eq 0 ]]; then
        # Start main service
        docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d data-indexer
    else
        # Start specific services
        docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d "${services[@]}"
    fi
    
    log_success "Services started"
}

# Function to stop services
stop_services() {
    log_info "Stopping services..."
    
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down
    
    log_success "Services stopped"
}

# Function to view logs
view_logs() {
    local service="${1:-data-indexer}"
    local follow="${2:-false}"
    
    if [[ "$follow" == "true" ]]; then
        docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" logs -f "$service"
    else
        docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" logs "$service"
    fi
}

# Function to run pipeline
run_pipeline() {
    # The first argument is the container name
    local container_name="$1"
    shift   # remove the first argument, so $@ now has the remaining args

    log_info "Running pipeline on container '$container_name' with arguments: $*"

    docker exec -it "$container_name" ./run.sh "$@"
}

# Function to get service status
get_status() {
    docker ps -f "name=$container_name"
}

# Function to access container shell
shell() {
    local service="${1:-data-indexer}"
    
    log_info "Accessing shell for service: $service"
    
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec "$service" /bin/bash
}

# Function to monitor resources
monitor() {
    local service="${1:-data-indexer}"
    
    log_info "Monitoring resources for service: $service"
    
    # Get container ID
    local container_id
    container_id=$(docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps -q "$service")
    
    if [[ -z "$container_id" ]]; then
        log_error "Service $service is not running"
        exit 1
    fi
    
    # Monitor resources
    echo "Monitoring container resources (Press Ctrl+C to stop)..."
    docker stats "$container_id"
}

# Function to cleanup
cleanup() {
    log_info "Cleaning up..."
    
    # Stop and remove containers
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down -v
    
    # Remove orphaned containers
    docker container prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    # Remove unused networks
    docker network prune -f
    
    log_success "Cleanup completed"
}

# Function to backup data
backup_data() {
    local backup_dir="${1:-./backups}"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_file="$backup_dir/data_backup_$timestamp.tar.gz"
    
    log_info "Creating backup..."
    
    mkdir -p "$backup_dir"
    
    # Backup logs and any persistent data
    tar -czf "$backup_file" logs/ config/ 2>/dev/null || true
    
    # Backup database if running
    if docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps postgres | grep -q "Up"; then
        log_info "Backing up database..."
        docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec -T postgres pg_dump -U postgres dataindex > "$backup_dir/db_backup_$timestamp.sql"
    fi
    
    log_success "Backup created: $backup_file"
}

# Function to restore data
restore_data() {
    local backup_file="$1"
    
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        exit 1
    fi
    
    log_info "Restoring from backup: $backup_file"
    
    # Extract backup
    tar -xzf "$backup_file" -C "$SCRIPT_DIR"
    
    log_success "Data restored from backup"
}

# Function to update configuration
update_config() {
    local lake_name="$1"
    local max_workers="${2:-}"
    local n_threads="${3:-}"
    
    if [[ -z "$lake_name" ]]; then
        log_error "Lake name is required"
        exit 1
    fi
    
    log_info "Updating configuration for lake: $lake_name"
    
    # Create or update lake config
    python3 -c "
import json
import os

config_file = 'lakes_config.json'
lake_name = '$lake_name'
max_workers = '$max_workers'
batch_size = '$batch_size'

# Load existing config or create new
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
else:
    config = {}

# Update configuration
if lake_name not in config:
    config[lake_name] = {}

if max_workers:
    config[lake_name]['max_workers'] = int(max_workers)
if batch_size:
    config[lake_name]['batch_size'] = int(batch_size)

# Save configuration
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print(f'Configuration updated for {lake_name}')
"
    
    log_success "Configuration updated"
}

# Function to show usage
usage() {
    cat << EOF
Docker Management Script for Data Lake Indexer

Usage: $0 <command> [options]

Commands:
    build                   Build the Docker image
    start [services...]     Start services (default: data-indexer)
    stop                    Stop all services
    restart [services...]   Restart services
    status                  Show service status
    logs [service] [-f]     View logs (use -f to follow)
    shell [service]         Access container shell
    monitor [service]       Monitor container resources
    run <args...>           Run the pipeline with arguments
    
    # Data management
    backup [dir]            Create backup (default: ./backups)
    restore <file>          Restore from backup
    cleanup                 Clean up containers, volumes, and networks
    
    # Configuration
    config <lake> [workers] [threads]  Update lake configuration
    
    # Examples
    $0 start                          # Start main service
    $0 start postgres                 # Start database only
    $0 run nyc gittables             # Run pipeline for specific lakes
    $0 run --from_checkpoint nyc     # Resume from checkpoint
    $0 logs data-indexer -f          # Follow logs
    $0 monitor                       # Monitor resources
    $0 config nyc 4 8               # Update nyc config

Environment Variables (set in .env file):
    MEMORY_LIMIT_GB         Memory limit for containers (default: 8)
    CPU_LIMIT              CPU limit for containers (default: 4)
    MAX_WORKERS            Maximum parallel workers (default: 2)
    N_THREADS              Number of threads per worker (default: 4)
    DATA_DIR               Host data directory path (required)
    DB_PASSWORD            Database password (required)

EOF
}

# Main script logic
main() {
    local command="${1:-}"
    
    if [[ -z "$command" ]]; then
        usage
        exit 1
    fi
    
    # Check Docker availability
    check_docker
    
    case "$command" in
        build)
            build_image
            ;;
        start)
            shift
            validate_env || exit 1
            start_services "$@"
            ;;
        stop)
            stop_services
            ;;
        restart)
            shift
            validate_env || exit 1
            stop_services
            start_services "$@"
            ;;
        status)
            get_status
            ;;
        logs)
            shift
            local service="${1:-data-indexer}"
            local follow="false"
            if [[ "${2:-}" == "-f" ]]; then
                follow="true"
            fi
            view_logs "$service" "$follow"
            ;;
        shell)
            shift
            shell "${1:-data-indexer}"
            ;;
        monitor)
            shift
            monitor "${1:-data-indexer}"
            ;;
        run)
            shift
            validate_env || exit 1
            run_pipeline "$@"
            ;;
        backup)
            shift
            backup_data "${1:-./backups}"
            ;;
        restore)
            shift
            if [[ -z "${1:-}" ]]; then
                log_error "Backup file path is required"
                exit 1
            fi
            restore_data "$1"
            ;;
        cleanup)
            cleanup
            ;;
        config)
            shift
            update_config "$@"
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"