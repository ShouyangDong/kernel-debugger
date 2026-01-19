# Auto CUDA Debugger

## Overview
Auto CUDA Debugger is a tool designed to automatically locate and debug errors in CUDA code, specifically for operations like multi-head attention (MHA) and gate MLP. The project provides a systematic approach to identify discrepancies between CUDA kernel outputs and expected results, allowing for efficient debugging and modification of the code.

## Features
- Load and run CUDA kernels for MHA and gate MLP.
- Analyze outputs from CUDA kernels and compare them with reference outputs.
- Automatically locate bad tokens and dimensions in the input data.
- Modify CUDA code based on analysis results to correct identified issues.
- Utility functions for logging, error handling, and data manipulation.

## Project Structure
```
auto-cuda-debugger
├── src
│   ├── find_bug.py        # Functions for loading libraries and finding discrepancies
│   ├── runner.py          # Executes the debugging process
│   ├── analyzer.py        # Analyzes kernel outputs and identifies discrepancies
│   ├── modifier.py        # Modifies CUDA code based on analysis results
│   ├── utils.py           # Utility functions for support
│   └── kernels
│       └── __init__.py    # Initializes the kernels package
├── kernels
│   ├── CMakeLists.txt     # Build configuration for CUDA kernels
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
   cd auto-cuda-debugger
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Build the CUDA kernels:
   ```
   cd kernels
   mkdir build
   cd build
   cmake ..
   make
   ```

## Usage
To use the Auto CUDA Debugger, run the `example_run.py` script located in the `examples` directory. This script demonstrates the workflow of locating errors and modifying the CUDA code.

```bash
python examples/example_run.py
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.