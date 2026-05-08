import os
import time
import psutil

def check_cpu_usage():
    """
    Checks the current CPU usage on the system.

    Returns:
        float: The percentage of CPU usage as a decimal (e.g. 0.8 means 80%)
    """
    # Get the total number of CPU cores
    num_cores = os.cpu_count()

    # Get the current CPU usage for each core
    cpu_usage = [psutil.cpu_percent(interval=1) / num_cores, psutil.cpu_percent(interval=1) / num_cores]

    # Calculate the total CPU usage as a percentage
    total_usage = sum(cpu_usage) * 100

    return total_usage

def check_disk_space():
    """
    Checks the available disk space on the system.

    Returns:
        float: The amount of available disk space in bytes as a float, or None if there are no physical drives
    """
    # Check for any physical drives
    drives = [drive for drive in os.listdir('/') if 'sd' in drive]
    if not drives:
        print("No physical drives detected.")
        return None

    # Get the total size of all drives on the system
    drive_sizes = [psutil.disk_usage(f'/dev/{drive}').total for drive in drives]

    # Calculate the available disk space as a percentage
    total_size = sum(drive_sizes)
    used_space = sum([psutil.disk_usage(f'/dev/{drive}').free for drive in drives])

    return (total_size - used_space) / total_size

def test_system_performance():
    """
    Tests the system performance by checking CPU, disk space, and memory usage.
    """
    print("Checking CPU usage...")
    cpu_usage = check_cpu_usage()
    print(f"CPU usage: {cpu_usage:.2f}%")

    print("Checking disk space...")
    disk_space = check_disk_space()
    if disk_space is not None:
        print(f"Disk space available: {disk_space * 10**-9:.4f} GB")
    else:
        print("Unable to determine disk space.")

    
if __name__ == "__main__":
    # Run the test system performance function
    test_system_performance()