#!/bin/bash

# Docker-optimized Pipeline Runner with Resource Management
# Usage: ./run_pipeline.sh [--from_checkpoint] lake1 lake2 ...

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/main.py"
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$SCRIPT_DIR/pipeline.pid"

# Resource limits (Docker-aware)
CPU_NICE=${CPU_NICE:-0}          # CPU priority (0-19, higher = lower priority)
IO_CLASS=${IO_CLASS:-2}           # I/O class: 1=RT, 2=best-effort, 3=idle
IO_PRIORITY=${IO_PRIORITY:-6}     # I/O priority (0-7, higher = lower priority)

if [[ -f /.dockerenv ]] || grep -q 'docker\|lxc' /proc/1/cgroup 2>/dev/null; then
    IS_DOCKER=true
else
    IS_DOCKER=false
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/runner.log"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/runner.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/runner.log"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/runner.log"
}

# Function to check if process is already running
check_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log_error "Pipeline is already running with PID $pid"
            exit 1
        else
            log_warn "Stale PID file found, removing..."
            rm -f "$PID_FILE"
        fi
    fi
}

# Docker-aware resource checking
check_resources() {
    log_info "Checking system resources..."
    log_info "Container environment: $IS_DOCKER"
    
    # Check memory usage
    local memory_usage
    if command -v free >/dev/null 2>&1; then
        memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
        if (( $(echo "$memory_usage > 85" | bc -l) )); then
            log_warn "High memory usage detected: ${memory_usage}%"
        fi
    else
        memory_usage="N/A"
    fi
    
    # Check disk space
    local disk_usage=$(df /mnt/data2 | tail -1 | awk '{print $5}' | sed 's/%//')
    if (( disk_usage > 95 )); then
        log_error "Low disk space: ${disk_usage}% used"
        exit 1
    fi
    
    # Check load average
    if [[ -f /proc/loadavg ]]; then
        local load_avg=$(cat /proc/loadavg | awk '{print $1}')
        log_info "Memory: ${memory_usage}%, Disk: ${disk_usage}%, Load: $load_avg"
    else
        log_info "Memory: ${memory_usage}%, Disk: ${disk_usage}%"
    fi
}

# Function to setup logging
setup_logging() {
    mkdir -p "$LOG_DIR"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    LOG_FILE="$LOG_DIR/pipeline_${timestamp}.log"
    ERROR_LOG="$LOG_DIR/pipeline_${timestamp}_error.log"
    RUNNER_LOG="$LOG_DIR/runner.log"
    
    log_info "Logs will be written to: $LOG_FILE"
}

# Enhanced resource monitoring for containers
monitor_resources() {
    local pid=$1
    local log_file=$2
    local resource_log="${log_file}.resources"
    
    log_info "Starting resource monitoring for PID $pid"
    
    while kill -0 "$pid" 2>/dev/null; do
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        local memory_mb="0"
        local cpu_percent="0"
        local container_memory_usage="N/A"
        local container_cpu_usage="N/A"
        
        # Process-specific resources
        if [[ -f "/proc/$pid/stat" ]]; then
            memory_mb=$(awk '{print $23/1024/1024}' "/proc/$pid/stat" 2>/dev/null || echo "0")
            cpu_percent=$(ps -p "$pid" -o %cpu= 2>/dev/null | awk '{print $1}' || echo "0")
        fi
        
        # Container-level resources (if available)
        if [[ "$IS_DOCKER" == "true" ]]; then
            if [[ -f /sys/fs/cgroup/memory/memory.usage_in_bytes ]]; then
                # cgroup v1
                local memory_usage_bytes=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo "0")
                container_memory_usage=$(echo "scale=1; $memory_usage_bytes / 1024 / 1024" | bc -l)
            elif [[ -f /sys/fs/cgroup/memory.current ]]; then
                # cgroup v2
                local memory_usage_bytes=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo "0")
                container_memory_usage=$(echo "scale=1; $memory_usage_bytes / 1024 / 1024" | bc -l)
            fi
            
            # CPU usage from cgroup
            if [[ -f /sys/fs/cgroup/cpuacct/cpuacct.usage ]]; then
                container_cpu_usage=$(cat /sys/fs/cgroup/cpuacct/cpuacct.usage 2>/dev/null || echo "0")
            elif [[ -f /sys/fs/cgroup/cpu.stat ]]; then
                container_cpu_usage=$(grep usage_usec /sys/fs/cgroup/cpu.stat 2>/dev/null | awk '{print $2}' || echo "0")
            fi
        fi
        
        echo "$timestamp - PID: $pid, Process Memory: ${memory_mb}MB, Process CPU: ${cpu_percent}%, Container Memory: ${container_memory_usage}MB, Container CPU: ${container_cpu_usage}" >> "$resource_log"
        
        # Check for memory pressure
        local memory_pressure=""
        if [[ -f /proc/pressure/memory ]]; then
            memory_pressure=$(grep "some avg10=" /proc/pressure/memory | awk -F= '{print $2}' | cut -d' ' -f1)
            if (( $(echo "$memory_pressure > 10" | bc -l 2>/dev/null || echo 0) )); then
                log_warn "Memory pressure detected: ${memory_pressure}%"
            fi
        fi
        
        sleep 60  # Log every minute
    done
    
    log_info "Resource monitoring stopped for PID $pid"
}

# Docker-aware process priorities
set_priorities() {
    local pid=$1
    
    # Set I/O priority (may not work in all container configurations)
    if command -v ionice &> /dev/null; then
        if ionice -c "$IO_CLASS" -n "$IO_PRIORITY" -p "$pid" 2>/dev/null; then
            log_info "Set I/O priority: class $IO_CLASS, priority $IO_PRIORITY"
        else
            log_warn "Failed to set I/O priority (container limitations or need sudo)"
        fi
    else
        log_warn "ionice not available"
    fi
    
    # CPU niceness
    if renice "$CPU_NICE" -p "$pid" 2>/dev/null; then
        log_info "Set CPU priority (nice): $CPU_NICE"
    else
        log_warn "Failed to set CPU priority"
    fi
}

# Enhanced cleanup function
cleanup() {
    log_info "Starting cleanup process..."
    
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            log_info "Terminating main process $pid..."
            
            # Try graceful shutdown first
            kill -TERM "$pid" 2>/dev/null || true
            
            # Wait for graceful shutdown
            local count=0
            while kill -0 "$pid" 2>/dev/null && [[ $count -lt 60 ]]; do
                if [[ $((count % 10)) -eq 0 ]]; then
                    log_info "Waiting for graceful shutdown... (${count}s)"
                fi
                sleep 1
                ((count++))
            done
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                log_warn "Force killing process $pid"
                kill -KILL "$pid" 2>/dev/null || true
                sleep 2
            fi
            
            # Clean up any child processes
            pkill -P "$pid" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
    fi
    
    # Clean up any Ray processes if they exist
    if command -v ray >/dev/null 2>&1; then
        ray stop 2>/dev/null || true
    fi
    
    # Final cleanup
    log_info "Cleanup completed"
}

# Signal handlers
trap cleanup EXIT INT TERM

# Main execution function
run_pipeline() {
    local from_checkpoint="$1"
    shift
    local lake_names=("$@")
    
    log_info "Starting pipeline with lakes: ${lake_names[*]}"
    if [[ "$from_checkpoint" == "true" ]]; then
        log_info "Resuming from checkpoint"
    fi
    log_info "Resource limits - CPU nice: $CPU_NICE, I/O class: $IO_CLASS, I/O priority: $IO_PRIORITY"
    
    # Build command arguments
    local cmd_args=()
    if [[ "$from_checkpoint" == "true" ]]; then
        cmd_args+=("--from_checkpoint")
    fi
    cmd_args+=("${lake_names[@]}")
    
    # Run the Python script with resource limits
    # nice -n "$CPU_NICE" python3 "$PYTHON_SCRIPT" "${cmd_args[@]}" \
    python3 "$PYTHON_SCRIPT" "${cmd_args[@]}" \
        > >(tee -a "$LOG_FILE") \
        2> >(tee -a "$ERROR_LOG" >&2) &
    
    local pid=$!
    echo "$pid" > "$PID_FILE"
    
    log_info "Started pipeline with PID: $pid"
    
    # Set process priorities
    set_priorities "$pid"
    
    # Start resource monitoring in background
    monitor_resources "$pid" "$LOG_FILE" &
    local monitor_pid=$!
    
    # Wait for the main process
    local exit_code=0
    if wait "$pid"; then
        log_success "Pipeline completed successfully"
    else
        exit_code=$?
        log_error "Pipeline failed with exit code: $exit_code"
    fi
    
    # Stop resource monitoring
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    
    rm -f "$PID_FILE"
    return $exit_code
}

# Main script logic
main() {
    # Parse arguments
    local from_checkpoint="false"
    local lake_names=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --from_checkpoint)
                from_checkpoint="true"
                shift
                ;;
            -*)
                log_error "Unknown option: $1"
                exit 1
                ;;
            *)
                lake_names+=("$1")
                shift
                ;;
        esac
    done
    
    # Check arguments
    if [[ ${#lake_names[@]} -lt 1 ]]; then
        echo "Usage: $0 [--from_checkpoint] <lake_name1> [lake_name2] ..."
        echo "Example: $0 lake1 lake2"
        echo "Example: $0 --from_checkpoint lake1 lake2"
        exit 1
    fi
    
    # Validate Python script exists
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        log_error "Python script not found: $PYTHON_SCRIPT"
        exit 1
    fi
    
    # Check for required commands
    for cmd in nice python3 bc; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "$cmd command not found"
            exit 1
        fi
    done
    
    # Check if already running
    check_running
    
    # Setup logging
    setup_logging
    
    # Check system resources
    check_resources
    
    # Run the pipeline
    run_pipeline "$from_checkpoint" "${lake_names[@]}"
}

# Run main function with all arguments
main "$@"