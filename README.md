# Price Level ML

This project aims to detect price levels in cryptocurrency trading data using machine learning techniques. The project fetches historical data from Bybit, processes it, and trains a model to identify significant price levels.

## Project Structure

- .env
- .gitignore
- data/
- processed/
- raw/
- mlruns/
- README.md
- requirements.txt
- src/


## Installation

    ```sh
    git clone https://github.com/yourusername/pricelevel-ml.git
    cd pricelevel-ml
    ```
    ```

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    ```

    ```sh
    pip install -r requirements.txt
    ```
    ```

## Configuration

Update the [.env](http://_vscodecontentref_/8) file with your Bybit API credentials and other configuration parameters.

## Usage

    ```sh
    python src/data_fetcher.py
    ```
    ```

    ```sh
    python src/main.py
    ```
    ```

## Project Modules

- **`src/config.py`**: Configuration parameters for the project.
- **`src/data_fetcher.py`**: Functions to fetch historical data from Bybit.
- **`src/dataset.py`**: Dataset class for preparing the data for training.
- **`src/cluster_levels.py`**: Functions to find and cluster local extrema in the price data.
- **`src/train.py`**: Functions to train and evaluate the model.
- **`src/main.py`**: Main script to run the entire pipeline.

## License
This project is licensed under the MIT License.
This project is licensed under the MIT License.