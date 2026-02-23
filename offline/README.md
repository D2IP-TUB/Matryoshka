# Data Lake Indexer - Docker Setup

This guide explains how to run the Data Lake Indexer in Docker containers with hard resource limits.

## Quick Start

1. **Clone and Setup**
   ```bash
   # Navigate to your project directory
   cd /path/to/your/project
   
   # Make scripts executable
   chmod +x run.sh docker-scripts.sh
   
   # Create and configure environment file
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Configure Environment**
   Edit the `.env` file with your settings:
   ```bash
   # Resource Limits
   MEMORY_LIMIT_GB=8        # Container memory limit
   CPU_LIMIT=4              # Container CPU limit
   MAX_WORKERS=2            # Parallel workers
   N_THREADS=4              # Threads per worker
   
   # Data Directory (must exist on host)
   DATA_DIR=/path/to/your/data
   
   # Database Configuration
   DB_PASSWORD=your_secure_password
   ```

3. **Build and Start**
   ```bash
   # Build the Docker image
   ./docker-scripts.sh build
   
   # Start services
   ./docker-scripts.sh start
   
   # Run the pipeline
   ./docker-scripts.sh run nyc gittables
   ```

## Resource Management

### Hard Resource Limits

The Docker Compose configuration enforces hard limits:

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'      # Maximum 4 CPU cores
      memory: 8G       # Maximum 8GB RAM
    reservations:
      cpus: '2.0'      # Reserved 2 CPU cores
      memory: 4G       # Reserved 4GB RAM
```

### Automatic Resource Detection

The updated `run.sh` script automatically detects container resource limits and adjusts:

- **CPU Detection**: Reads from cgroup limits or environment variables
- **Memory Detection**: Reads from cgroup limits or environment variables
- **Worker Scaling**: Automatically scales `MAX_WORKERS` and `N_THREADS` based on available resources

### Resource Monitoring

Monitor resource usage in real-time:

```bash
# Monitor container resources
./docker-scripts.sh monitor

# View detailed resource logs
./docker-scripts.sh logs data-indexer -f
```

## Configuration Files

### Environment Variables (.env)

```bash
# Resource Limits
MEMORY_LIMIT_GB=8          # Container memory limit in GB
CPU_LIMIT=4                # Container CPU core limit
MAX_WORKERS=2              # Maximum parallel workers
N_THREADS=4                # Threads per worker

# Data Configuration
DATA_DIR=/path/to/data     # Host path to data directory
SSH_TUNNEL=false           # Enable SSH tunneling if needed

# Database Configuration
DB_HOST=postgres           # Database host (use service name for containers)
DB_PORT=5432              # Database port
DB_NAME=dataindex         # Database name
DB_USER=postgres          # Database user
DB_PASSWORD=secure_pass   # Database password (change this!)

# Logging
LOG_LEVEL=INFO            # Logging level
JSON_LOGGING=false        # Use JSON format for logs
```

### Lakes Configuration (lakes_config.json)

The container automatically adjusts lake configurations based on resource limits:

```json
{
  "nyc": {
    "data_dir": "/app/data/nyc",
    "feature_selection_table_name": "NYC_matryoshka_fs_index_key_oriented",
    "overlap_table_name": "NYC_matryoshka_overlap_index_key_oriented",
    "n_threads": 4,
    "max_workers": 2,
    "from_tar_archive": true,
    "feature_extraction": false
  }
}
```

## Usage Examples

### Basic Operations

```bash
# Build and start
./docker-scripts.sh build
./docker-scripts.sh start

# Check status
./docker-scripts.sh status

# View logs
./docker-scripts.sh logs data-indexer
./docker-scripts.sh logs data-indexer -f  # Follow logs

# Access container shell
./docker-scripts.sh shell
```

### Running Pipelines

```bash
# Run specific lakes
./docker-scripts.sh run nyc gittables

# Resume from checkpoint
./docker-scripts.sh run --from_checkpoint nyc

# Run with custom configuration
./docker-scripts.sh run --from_checkpoint gittables cuk
```

### Resource Monitoring

```bash
# Monitor container resources
./docker-scripts.sh monitor data-indexer

# View resource usage logs
tail -f logs/pipeline_*_resources

# Check container limits
docker inspect data-lake-indexer | grep -A 20 "Resources"
```

### Database Operations

```bash
# Start with database
./docker-scripts.sh start data-indexer postgres

# Access database
docker-compose exec postgres psql -U postgres -d dataindex

# Backup database
./docker-scripts.sh backup ./backups

# View database logs
./docker-scripts.sh logs postgres
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```bash
   # Check memory usage
   ./docker-scripts.sh monitor
   
   # Reduce workers in .env
   MAX_WORKERS=1
   N_THREADS=2
   
   # Restart with new limits
   ./docker-scripts.sh restart
   ```

2. **CPU Limit Reached**
   ```bash
   # Check CPU usage
   docker stats data-lake-indexer
   
   # Reduce CPU-intensive operations
   CPU_LIMIT=2
   MAX_WORKERS=1
   ```

3. **Disk Space Issues**
   ```bash
   # Check disk usage
   df -h
   
   # Clean up Docker resources
   ./docker-scripts.sh cleanup
   ```

4. **Permission Issues**
   ```bash
   # Fix file permissions
   sudo chown -R $(id -u):$(id -g) logs/ data/
   
   # Check SELinux (if applicable)
   sestatus
   ```

### Debugging

```bash
# Access container shell for debugging
./docker-scripts.sh shell

# Check container logs
./docker-scripts.sh logs data-indexer

# View resource usage
cat logs/pipeline_*_resources

# Check cgroup limits inside container
./docker-scripts.sh shell
cat /sys/fs/cgroup/memory/memory.limit_in_bytes
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us
```

## Advanced Configuration

### Custom Resource Limits

Modify `docker-compose.yml` for different resource requirements:

```yaml
deploy:
  resources:
    limits:
      cpus: '8.0'      # 8 CPU cores
      memory: 16G      # 16GB RAM
    reservations:
      cpus: '4.0'      # 4 CPU cores reserved
      memory: 8G       # 8GB RAM reserved
```

### Multiple Lake Processing

```bash
# Process multiple lakes with different configurations
./docker-scripts.sh config nyc 4 8      # 4 workers, 8 threads
./docker-scripts.sh config gittables 2 4 # 2 workers, 4 threads
./docker-scripts.sh run nyc gittables
```

### Monitoring Integration

Enable monitoring services:

```bash
# Start with monitoring
docker-compose --profile monitoring up -d

# Access monitoring
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

## Performance Tuning

### Memory Optimization

1. **Container Level**:
   - Set appropriate `memory` limits
   - Use `memswap_limit` to control swap usage
   - Monitor memory pressure with `/proc/pressure/memory`

2. **Application Level**:
   - Reduce `MAX_WORKERS` for memory-intensive operations
   - Use `ray` memory limits: `ray.init(object_store_memory=2000000000)`
   - Enable garbage collection optimizations

### CPU Optimization

1. **Container Level**:
   - Set CPU limits with `cpus`
   - Use CPU affinity with `cpuset`
   - Adjust CPU shares with `cpu_shares`

2. **Application Level**:
   - Balance `MAX_WORKERS` and `N_THREADS`
   - Use CPU-aware scheduling
   - Monitor CPU pressure

### I/O Optimization

1. **Container Level**:
   - Use appropriate storage drivers
   - Mount data volumes with optimized options
   - Configure I/O limits with `blkio`

2. **Application Level**:
   - Use ionice for I/O priority
   - Implement batched I/O operations
   - Monitor disk pressure

## Security Considerations

1. **Container Security**:
   - Run as non-root user
   - Use `no-new-privileges` security option
   - Limit system capabilities

2. **Network Security**:
   - Use custom networks
   - Expose only necessary ports
   - Implement proper firewall rules

3. **Data Security**:
   - Use read-only mounts for source data
   - Secure database credentials
   - Implement backup encryption

## Maintenance

### Regular Tasks

```bash
# Check system health
./docker-scripts.sh status
./docker-scripts.sh monitor

# Create backups
./docker-scripts.sh backup

# Clean up resources
./docker-scripts.sh cleanup

# Update images
docker-compose pull
./docker-scripts.sh build
```

### Log Management

```bash
# Rotate logs
docker-compose exec data-indexer logrotate -f /etc/logrotate.conf

# Archive old logs
tar -czf logs_archive_$(date +%Y%m%d).tar.gz logs/

# Clean up old logs
find logs/ -name "*.log" -mtime +7 -delete
```