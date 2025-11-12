# Installation and Build

1.  **Requirements**:
    - A C++20 compliant compiler (e.g., modern GCC, Clang).
    - CMake (version 3.18+).
    - Python (3.8+).
    - `pybind11`, `numpy`, `scipy`  Python packages.
    - For macOS with OpenMP, `llvm` is required (`brew install llvm`).

2.  **Build Steps**:
    From the project root, run the following commands:

    ```bash
    # (Activate your virtual environment first)
    #pip install pybind11 numpy scipy

    #
    CC=/opt/homebrew/opt/llvm/bin/clang CXX=/opt/homebrew/opt/llvm/bin/clang++ pip install -e .
    ```

# Test 

in $ROOT_CLIC_CLIB/tests/

```bash
bash run_tests.sh
```

# Usage 

See the examples in tests/