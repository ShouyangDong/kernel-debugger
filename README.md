# Auto Debugger

## Overview
Auto Debugger (Automatic Delta Debugger) is a built-in debugging tool for Xpiler that automatically sompilifies complex Python programs to the minimal code needed to reproduce a specific error. This is extremely useful for debugging large, complex Xpiler programs.

## What is Delta Debugging?

Delta Debugging is an automated debugging technique with the core idea:
1. Given a program that triggers a bug.
2. Systematically remove code fragments from the program.
3. Check if the simplified program still triggers the same bug.
4. Eventually obtain the minimal code that triggers the bug.

Auto Delta Debugger uses a Probability Driven Delta Debugging (PDD) algorithm for efficient search of minimized code.
## Why Auto Debugger?

When transcompiling tensor programs, bugs are often hidden in complex code.

- **Lost of irrelevent code**: Real projects may have hundreds of lines of configuration, helper functions, logging, etc.
- **Hard to locate**: Error messages may point to underlying CUDA code.
- **Tedious debugging**: Manually deleting code locate bugs is very time-consuming.



## Usage

### Basic Usage

```
python -m src.auto_debug <source_file> --error-msg <error_message>" -o <output_file>
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `source`  |Path to the input Python source file|
|`--err-msg`|Error message to match (searched in stdout or stderr)
|`-o, --output`| Path to the minimized output file|
|`--backend`| Execution backend: `runner` (faster) or `subproc` (more stable), default `runner`|
|`--timeout`| Timeout for each task in seconds, default 60|
|`-j, --jobs`| Number of parallel jobs, default 1


### Example

Run Auto Debugger on `xpiler_buggy.py` in this directory:

```bash
# Use 4 parallel jobs, search for "Dimension mismatch" error
python -m src.auto_debug xpiler_buggy.py --err-msg "Dimension mismatch" -o minimized.py -j 4

# Or use subprocess backend (more stable but slower)
python -m src.auto_debug xpiler_buggy.py --err-msg "Dimension mismatch" -o minimized.py --backend subproc
```


## Features
- Load and run kernels for MHA and gate MLP.
- Analyze outputs from kernels and compare them with reference outputs.
- Automatically locate bad tokens and dimensions in the input data.
- Modify code based on analysis results to correct identified issues.
- Utility functions for logging, error handling, and data manipulation.

## Project Structure
```
auto-debugger
├── src
│   ├── find_bug.py        # Functions for loading libraries and finding discrepancies
│   ├── runner.py          # Executes the debugging process
│   ├── analyzer.py        # Analyzes kernel outputs and identifies discrepancies
│   ├── modifier.py        # Modifies code based on analysis results
│   ├── utils.py           # Utility functions for support
│   └── kernels
│       └── __init__.py    # Initializes the kernels package
├── kernels
│   ├── CMakeLists.txt     # Build configuration for kernels
│   ├── mha.cu             # Implementation of the multi-head attention kernel
│   └── gatemlp.cu         # Implementation of the gate MLP kernel
├── tests
│   └── test_find_bug.py   # Unit tests for find_bug.py
├── examples
│   └── example_run.py     # Example usage of the debugging tools
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
└── README.md              # Project documentation
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd auto-debugger
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Build the kernels:
   ```
   cd kernels
   mkdir build
   cd build
   cmake ..
   make
   ```

## Usage
To use the Auto Debugger, run the `example_run.py` script located in the `examples` directory. This script demonstrates the workflow of locating errors and modifying the code.

```bash
python examples/example_run.py
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.