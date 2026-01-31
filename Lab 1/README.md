# Lab 1: MLOps Foundation - Calculator & CI/CD

## Objective
This lab focuses on the fundamental practices of MLOps, including environment isolation, structured development, automated testing, and continuous integration.

## Unique Features (Differentiated Submission)
To go beyond the basic requirements, this implementation includes:
- **Advanced Math Functions**: Power ($x^y$), Square $Root$ ($\sqrt{x}$ with validation), and List $Average$ functions.
- **Dual Testing Strategy**: Full test suites implemented and verified in both `pytest` and `unittest`.
- **Automated CI/CD**: Two independent GitHub Actions workflows to ensure all tests pass on every code change.

## Project Structure
- `src/`: Core calculator logic.
- `test/`: Comprehensive test suites.
- `data/`: Placeholder for future datasets.
- `.github/workflows/`: Automation pipelines.

## How to Run Locally

### 1. Setup Environment
```bash
# Navigate to Lab 1
cd "Lab 1"

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Tests
- **Using Pytest**:
  ```bash
  pytest test/test_pytest.py
  ```
- **Using Unittest**:
  ```bash
  python3 -m unittest test/test_unittest.py
  ```
