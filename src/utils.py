def log_message(message):
    with open("debug_log.txt", "a") as log_file:
        log_file.write(message + "\n")

def handle_error(error):
    log_message(f"Error: {error}")

def validate_input(data, expected_shape):
    if data.shape != expected_shape:
        handle_error(f"Input shape {data.shape} does not match expected shape {expected_shape}.")
        return False
    return True

def save_debug_info(info, filename="debug_info.txt"):
    with open(filename, "w") as file:
        file.write(info)

def load_debug_info(filename="debug_info.txt"):
    try:
        with open(filename, "r") as file:
            return file.read()
    except FileNotFoundError:
        handle_error(f"{filename} not found.")
        return None

def print_debug_info(info):
    print(info)