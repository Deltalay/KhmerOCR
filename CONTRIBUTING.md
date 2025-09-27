# Contributing to KhmerOCR

Welcome! This document outlines how to contribute to this project, the folder structure, coding principles, and best practices to follow. Following these guidelines helps maintain a clean, readable, and maintainable codebase.


## Coding Principles (Linux Kernel Inspired)

### 1. Simplicity
- Keep the code simple and intuitive.
- Avoid overly clever or obscure Python tricks.
- Example: Use descriptive class and function names.

### 2. Readability
- Code should be understandable by humans first.
- Use type hints and docstrings for all classes/functions.
- Maintain consistent formatting with tools like `black` or `flake8`.

### 3. Modularity
- Separate components: CNN, ViT, data, training, evaluation.
- Each module should have a single responsibility.
- Reuse modules instead of copying code.

### 4. Maintainability
- Use configuration files (`YAML`/`JSON`) for hyperparameters.
- Write small, testable functions.
- Log metrics properly (e.g., TensorBoard, wandb).

### 5. Minimalism
- Only implement features that are necessary.
- Avoid over-engineering or adding unused components.

### 6. Documentation
- Document tricky sections and architectural decisions.
- Keep README and inline comments up-to-date.

## Coding Guidelines

- **Naming Conventions**
  - Classes: `PascalCase` (e.g., `CNNEncoder`)
  - Functions/variables: `snake_case` (e.g., `load_dataset`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `LEARNING_RATE`)
  
- **Imports**
  - Standard library first, then third-party, then local modules.
  
- **Type Hints**
  - All functions should include type hints wherever applicable.

- **Docstrings**
  - Use Google-style or NumPy-style docstrings.
  
- **Version Control**
  - Write clear, concise commit messages.



## How to Contribute

1. Fork the repository.
2. Create a branch for your feature or bugfix.
3. Implement changes following the coding principles.
4. Run existing tests and add new tests if needed.
5. Submit a Pull Request with a clear description of changes.

---

By following these guidelines, you help ensure that this project remains **clean, modular, maintainable, and readable**, just like a well-engineered kernel—but for deep learning!  

