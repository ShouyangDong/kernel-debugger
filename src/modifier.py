import os


def modify_cuda_code(kernel_file, modifications):
    with open(kernel_file, "r") as file:
        code = file.readlines()

    for line_number, new_line in modifications.items():
        if 0 <= line_number < len(code):
            code[line_number] = new_line + "\n"

    with open(kernel_file, "w") as file:
        file.writelines(code)


def adjust_parameters(kernel_file, param_name, new_value):
    modifications = {}
    with open(kernel_file, "r") as file:
        code = file.readlines()

    for i, line in enumerate(code):
        if param_name in line:
            parts = line.split("=")
            if len(parts) == 2:
                modifications[i] = f"{param_name} = {new_value}"

    modify_cuda_code(kernel_file, modifications)


def example_modification():
    # Example for multi-head attention kernel
    kernel_file = os.path.join("kernels", "mha.cu")
    modifications = {
        10: "__shared__ float shared_memory[1024];",  # Example modification
    }
    modify_cuda_code(kernel_file, modifications)

    # Example parameter adjustment
    adjust_parameters(kernel_file, "BLOCK_SIZE", 128)


if __name__ == "__main__":
    example_modification()
