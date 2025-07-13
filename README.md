
# PyMineOptions

A Python framework and graphical user interface (GUI) for the valuation of mining project option pricing, leveraging Monte Carlo simulation and real options analysis. This tool provides a robust and flexible approach to value investment projects with embedded flexibility, affected by multiple statistically independent uncertainties.

---

## Table of Contents
- [Features](#features)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [About the Developer](#about-the-developer)
- [License](#license)


## Features
- **Modular Design:** Separated into two core modules for clear functionality:
  - **Module 1:** Generates output-sigmas of project value log-returns, measuring the effect of individual input risk variables on volatility. Includes flexible probability distributions for input variables (Triangular, Gaussian).
  - **Module 2:** Builds and rolls back decision trees (lattices) to obtain project option value based on volatilities from Module 1, supporting both simple and compound options.
- **Monte Carlo Simulation:** Quantifies the impact of input risk variables on project value volatility.
- **Real Options Valuation:** Implements recursive binomial lattice models for valuing real options, including sequential or compound options.
- **Intuitive GUI:** User-friendly graphical interface for selecting input risk variables and executing simulations.
- **External Data Integration:** Loads fixed project parameters from external `.mat` files.

---

## Getting Started
Follow these steps to get PyMineOptions up and running on your local machine.

pip (Python package installer)
git clone <https://github.com/pajoomand/PyMineOptions> # If using Git
pip install numpy scipy
numpy: Essential for numerical operations and array handling.
tkinter and ttk are typically included with Python installations.

## Prerequisites
- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/) (Python package installer)

## Installation

1. **Clone or Download the Project:**
   ```sh
   git clone <https://github.com/pajoomand/PyMineOptions>
   # Or download the ZIP and extract it
   ```

2. **Navigate to the Project Directory:**
   ```sh
   cd PyMineOptions
   ```

3. **(Recommended) Create a Virtual Environment:**
   ```sh
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On macOS/Linux
   ```

4. **Install Required Python Libraries:**
   ```sh
   pip install numpy scipy
   ```
   Or use a `requirements.txt` file:
   ```sh
   pip install -r requirements.txt
   ```

   - `numpy`: Essential for numerical operations and array handling.
   - `scipy`: Provides scientific computing tools, including `scipy.ndimage.interpolation.shift` and `scipy.io.loadmat`.
   - `tkinter` and `ttk` are typically included with Python installations.

5. **Prepare Data File:**
   - Ensure your `cashflow_fixed_parameters.mat` file is placed in the `data/` directory relative to the main project folder.
   - Example path: `PyMineOptions/data/cashflow_fixed_parameters.mat`



## Usage
You can use PyMineOptions in two ways:

### 1. Run the GUI Application

1. Open a Terminal or Command Prompt in the `PyMineOptions/` root directory.
2. Execute the GUI script:
   ```sh
   python PyMineOptions_GUI.py
   ```
3. The GUI window will appear, allowing you to:
   - Select one or more input risk variables (ROM, Yield, Exchange Rate, Price) using checkboxes.
   - Choose the number of Monte Carlo iterations from the dropdown.
   - Click the **Simulate** button to calculate and display the output sigma for the selected variables.

### 2. Run Example Scripts

You can also run the following scripts for example usage of the framework:

- `module1_script.py`: Demonstrates usage of Module 1 (volatility and risk analysis).
- `module2_script.py`: Demonstrates usage of Module 2 (real options lattice models).

Run them from the project root directory:
```sh
python module1_script.py
python module2_script.py
```


## Project Structure
The project is organized into logical modules and folders:

```
PyMineOptions/
├── .vscode/                     # VS Code specific configuration
├── __pycache__/                 # Python generated cache files
├── data/                        # Contains external data files
│   └── cashflow_fixed_parameters.mat # Fixed project parameters
├── module1/                     # Core financial modeling components
│   ├── __init__.py              # Makes 'module1' a Python package
│   ├── cash_flow.py             # Defines the Prototype_Cash_flow model
│   ├── risk_parameters.py       # Defines risk variables
│   ├── convert_normal_to_lognormal.py # Utility for distribution conversion
│   ├── distribution.py          # Defines the Distribution class
│   ├── model_setup.py           # Models the cash flow for a mining project
│   ├── fund_info.py             # Defines the Fund_info class
│   ├── generate_output_sigma.py # Defines the generate_output_sigma class for MC
├── module2/                     # Real options lattice models
│   ├── __init__.py              # Makes 'module2' a Python package
│   ├── Maximisation_class.py    # Defines parameters for option exercise
│   └── real_options_lattice.py  # Defines RecursiveLattice, SimpleRecursiveLattice, CompoundRecursiveLattice
├── create_Prototype_Cash_flow.py # Helper function to instantiate the Prototype Cash Flow model
├── module1_script.py            # Example script demonstrating Module 1 usage
├── module2_script.py            # Example script demonstrating Module 2 usage
└── PyMineOptions_GUI.py         # The main graphical user interface application
```

---

## Troubleshooting

- **tkinter not found:** On some minimal Python installations, `tkinter` may not be included. Install it via your package manager (e.g., `sudo apt-get install python3-tk` on Ubuntu) or ensure your Python installation includes it.
- **.mat file errors:** Ensure `cashflow_fixed_parameters.mat` is present in the correct `data/` directory.
- **Permission issues:** Run your terminal or command prompt as administrator if you encounter permission errors.


## About the Developer
**Tom Pazoum (H. Pazhoumanddar)**

- Email: tom.pazoum@gmail.com
- [LinkedIn](https://www.linkedin.com/in/tom-pazoum-59b62070/)
- [GitHub](https://github.com/pajoomand)

---

## License
This project is open-source and available under the MIT License. See the [LICENSE](LICENSE) file for details.