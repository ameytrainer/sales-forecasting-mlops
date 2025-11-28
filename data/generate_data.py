#!/usr/bin/env python3
"""
=====================================================
GENERATE_DATA.PY
Synthetic Sales Data Generator for MLOps Training
=====================================================
Author: Amey Talkatkar
Email: ameytalkatkar169@gmail.com
GitHub: https://github.com/ameytrainer
Course: MLOps with Agentic AI - Advanced Certification

Purpose:
    Generate realistic synthetic sales data for training MLOps pipeline.
    Includes seasonality, trends, regional patterns, and realistic noise.

Usage:
    python data/generate_data.py --rows 10000 --output data/raw/sales_data.csv
    python data/generate_data.py --start-date 2023-01-01 --end-date 2024-12-31
    python data/generate_data.py --format parquet --output data/raw/sales_data.parquet

Features:
    - Realistic seasonality (Q4 boost, summer patterns)
    - Day-of-week effects (weekend vs weekday)
    - Regional preferences
    - Product category patterns
    - Price variations
    - Quantity distributions
    - Time-based trends
    - Configurable parameters

Last Updated: November 24, 2024
=====================================================
"""

import argparse
import logging
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Optional: For faster Parquet writing
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    logging.warning("PyArrow not available. Parquet format will be slower.")

# =====================================================
# CONFIGURATION & CONSTANTS
# =====================================================

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Regional configuration
REGIONS = ["North", "South", "East", "West"]
REGION_POPULATIONS = {
    "North": 0.30,  # 30% of sales
    "South": 0.25,  # 25% of sales
    "East": 0.25,   # 25% of sales
    "West": 0.20,   # 20% of sales
}

# Product configuration
PRODUCTS = {
    "Electronics": {
        "category": "Technology",
        "base_price": 299.99,
        "price_std": 150.0,
        "min_price": 49.99,
        "max_price": 999.99,
        "base_quantity": 10,
        "quantity_std": 5,
        "seasonality": "q4",  # High in Q4 (holidays)
        "weekend_boost": 1.2,  # 20% more on weekends
    },
    "Clothing": {
        "category": "Apparel",
        "base_price": 49.99,
        "price_std": 30.0,
        "min_price": 9.99,
        "max_price": 199.99,
        "base_quantity": 25,
        "quantity_std": 15,
        "seasonality": "summer_winter",  # High in summer and winter
        "weekend_boost": 1.5,  # 50% more on weekends
    },
    "Food": {
        "category": "Groceries",
        "base_price": 12.99,
        "price_std": 8.0,
        "min_price": 2.99,
        "max_price": 49.99,
        "base_quantity": 50,
        "quantity_std": 30,
        "seasonality": "stable",  # Stable throughout year
        "weekend_boost": 1.1,  # 10% more on weekends
    },
    "Home": {
        "category": "Furniture",
        "base_price": 199.99,
        "price_std": 100.0,
        "min_price": 29.99,
        "max_price": 499.99,
        "base_quantity": 8,
        "quantity_std": 4,
        "seasonality": "spring",  # High in spring (home improvement)
        "weekend_boost": 1.3,  # 30% more on weekends
    },
    "Sports": {
        "category": "Outdoor",
        "base_price": 79.99,
        "price_std": 50.0,
        "min_price": 14.99,
        "max_price": 299.99,
        "base_quantity": 15,
        "quantity_std": 10,
        "seasonality": "summer",  # High in summer
        "weekend_boost": 1.4,  # 40% more on weekends
    },
}

# Regional product preferences (which regions prefer which products)
REGIONAL_PREFERENCES = {
    "North": {"Electronics": 1.2, "Clothing": 1.1, "Food": 1.0, "Home": 0.9, "Sports": 0.8},
    "South": {"Electronics": 0.9, "Clothing": 1.2, "Food": 1.1, "Home": 1.0, "Sports": 1.3},
    "East": {"Electronics": 1.3, "Clothing": 1.0, "Food": 1.0, "Home": 1.1, "Sports": 0.9},
    "West": {"Electronics": 1.1, "Clothing": 1.3, "Food": 0.9, "Home": 1.2, "Sports": 1.1},
}

# Seasonality multipliers by month
SEASON_MULTIPLIERS = {
    1: 0.8,   # January (post-holiday slowdown)
    2: 0.85,  # February
    3: 0.95,  # March
    4: 1.0,   # April (spring)
    5: 1.05,  # May
    6: 1.1,   # June (summer begins)
    7: 1.15,  # July (summer peak)
    8: 1.1,   # August
    9: 1.0,   # September (back to school)
    10: 1.05, # October
    11: 1.2,  # November (Black Friday)
    12: 1.4,  # December (holidays)
}

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def get_season(month: int) -> str:
    """
    Get season name from month number.
    
    Args:
        month: Month number (1-12)
    
    Returns:
        Season name (Spring, Summer, Fall, Winter)
    """
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:  # 12, 1, 2
        return "Winter"


def get_seasonality_multiplier(product: str, month: int) -> float:
    """
    Get seasonality multiplier for a product in a given month.
    
    Args:
        product: Product name
        month: Month number (1-12)
    
    Returns:
        Multiplier (e.g., 1.3 = 30% boost)
    """
    seasonality_type = PRODUCTS[product]["seasonality"]
    base_multiplier = SEASON_MULTIPLIERS[month]
    
    if seasonality_type == "q4":
        # Electronics: Boost in Q4
        if month in [11, 12]:
            return base_multiplier * 1.5
        elif month in [1, 2]:
            return base_multiplier * 0.7
        return base_multiplier
    
    elif seasonality_type == "summer":
        # Sports: Boost in summer
        if month in [6, 7, 8]:
            return base_multiplier * 1.4
        elif month in [12, 1, 2]:
            return base_multiplier * 0.6
        return base_multiplier
    
    elif seasonality_type == "summer_winter":
        # Clothing: Boost in summer and winter
        if month in [6, 7, 8, 12, 1, 2]:
            return base_multiplier * 1.3
        return base_multiplier
    
    elif seasonality_type == "spring":
        # Home: Boost in spring
        if month in [3, 4, 5]:
            return base_multiplier * 1.4
        return base_multiplier
    
    else:  # stable
        # Food: Relatively stable
        return base_multiplier


def get_day_of_week_multiplier(day_of_week: int, product: str) -> float:
    """
    Get day-of-week multiplier for sales.
    
    Args:
        day_of_week: Day of week (0=Monday, 6=Sunday)
        product: Product name
    
    Returns:
        Multiplier for sales quantity
    """
    is_weekend = day_of_week >= 5  # Saturday or Sunday
    
    if is_weekend:
        return PRODUCTS[product]["weekend_boost"]
    else:
        # Weekdays: slight variation
        weekday_multipliers = {
            0: 0.95,  # Monday (slow start)
            1: 1.0,   # Tuesday
            2: 1.0,   # Wednesday
            3: 1.05,  # Thursday
            4: 1.1,   # Friday (preparing for weekend)
        }
        return weekday_multipliers[day_of_week]


def generate_price(product: str) -> float:
    """
    Generate a realistic price for a product.
    
    Args:
        product: Product name
    
    Returns:
        Price (with some random variation)
    """
    config = PRODUCTS[product]
    
    # Normal distribution around base price
    price = np.random.normal(config["base_price"], config["price_std"])
    
    # Clip to min/max range
    price = np.clip(price, config["min_price"], config["max_price"])
    
    # Round to 2 decimal places (like real prices)
    # Add some "psychological pricing" (*.99)
    if random.random() < 0.7:  # 70% chance of .99 pricing
        price = np.floor(price) + 0.99
    else:
        price = np.round(price, 2)
    
    return price


def generate_quantity(
    product: str,
    region: str,
    month: int,
    day_of_week: int,
    trend_multiplier: float = 1.0
) -> int:
    """
    Generate realistic sales quantity with multiple factors.
    
    Args:
        product: Product name
        region: Region name
        month: Month number (1-12)
        day_of_week: Day of week (0-6)
        trend_multiplier: Overall trend multiplier
    
    Returns:
        Quantity sold
    """
    config = PRODUCTS[product]
    
    # Base quantity
    base_qty = config["base_quantity"]
    
    # Apply seasonality
    seasonal_mult = get_seasonality_multiplier(product, month)
    
    # Apply day-of-week effect
    dow_mult = get_day_of_week_multiplier(day_of_week, product)
    
    # Apply regional preference
    regional_mult = REGIONAL_PREFERENCES[region][product]
    
    # Apply overall trend
    trend_mult = trend_multiplier
    
    # Calculate expected quantity
    expected_qty = base_qty * seasonal_mult * dow_mult * regional_mult * trend_mult
    
    # Add random variation (Poisson distribution for count data)
    quantity = np.random.poisson(expected_qty)
    
    # Ensure at least 1 item sold
    quantity = max(1, quantity)
    
    return int(quantity)


def calculate_trend_multiplier(date: datetime, start_date: datetime, end_date: datetime) -> float:
    """
    Calculate overall trend multiplier (growth over time).
    
    Simulates business growth of 15% per year.
    
    Args:
        date: Current date
        start_date: Dataset start date
        end_date: Dataset end date
    
    Returns:
        Trend multiplier
    """
    total_days = (end_date - start_date).days
    current_days = (date - start_date).days
    
    # 15% annual growth
    annual_growth = 0.15
    
    # Calculate growth based on position in timeline
    years_elapsed = current_days / 365.25
    multiplier = 1.0 + (annual_growth * years_elapsed)
    
    return multiplier


def generate_sales_record(
    date: datetime,
    start_date: datetime,
    end_date: datetime
) -> Dict:
    """
    Generate a single sales record with realistic patterns.
    
    Args:
        date: Transaction date
        start_date: Dataset start date
        end_date: Dataset end date
    
    Returns:
        Dictionary with sales record
    """
    # Select region (weighted by population)
    region = random.choices(
        list(REGIONS),
        weights=list(REGION_POPULATIONS.values())
    )[0]
    
    # Select product (uniform for now, could add weights)
    product = random.choice(list(PRODUCTS.keys()))
    category = PRODUCTS[product]["category"]
    
    # Date-based features
    month = date.month
    day_of_week = date.weekday()  # 0=Monday, 6=Sunday
    is_weekend = day_of_week >= 5
    season = get_season(month)
    
    # Calculate trend multiplier
    trend_mult = calculate_trend_multiplier(date, start_date, end_date)
    
    # Generate price and quantity
    price = generate_price(product)
    quantity = generate_quantity(product, region, month, day_of_week, trend_mult)
    
    # Calculate total sales
    sales = price * quantity
    
    # Create record
    record = {
        "date": date.strftime("%Y-%m-%d"),
        "region": region,
        "product": product,
        "category": category,
        "price": round(price, 2),
        "quantity": quantity,
        "sales": round(sales, 2),
        "month": month,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "season": season,
    }
    
    return record


def generate_dataset(
    num_rows: int,
    start_date: datetime,
    end_date: datetime,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Generate complete sales dataset.
    
    Args:
        num_rows: Number of records to generate
        start_date: Start date for data
        end_date: End date for data
        show_progress: Whether to show progress updates
    
    Returns:
        DataFrame with sales data
    """
    logger.info(f"Generating {num_rows:,} sales records...")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Generate date range
    total_days = (end_date - start_date).days
    dates = [start_date + timedelta(days=random.randint(0, total_days)) for _ in range(num_rows)]
    dates.sort()  # Sort by date for realism
    
    # Generate records
    records = []
    for i, date in enumerate(dates):
        record = generate_sales_record(date, start_date, end_date)
        records.append(record)
        
        # Progress update
        if show_progress and (i + 1) % 1000 == 0:
            logger.info(f"Generated {i + 1:,} / {num_rows:,} records ({(i+1)/num_rows*100:.1f}%)")
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Convert data types
    df["date"] = pd.to_datetime(df["date"])
    df["region"] = df["region"].astype("category")
    df["product"] = df["product"].astype("category")
    df["category"] = df["category"].astype("category")
    df["season"] = df["season"].astype("category")
    df["is_weekend"] = df["is_weekend"].astype(bool)
    
    logger.info(f"✅ Dataset generated successfully: {len(df):,} rows, {len(df.columns)} columns")
    
    return df


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Validate generated dataset for quality issues.
    
    Args:
        df: Generated DataFrame
    
    Returns:
        True if valid, raises exception if invalid
    """
    logger.info("Validating dataset...")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        raise ValueError(f"Dataset has {missing} missing values!")
    
    # Check date range
    min_date = df["date"].min()
    max_date = df["date"].max()
    logger.info(f"Date range: {min_date.date()} to {max_date.date()}")
    
    # Check numeric ranges
    assert df["price"].min() > 0, "Price should be positive"
    assert df["quantity"].min() > 0, "Quantity should be positive"
    assert df["sales"].min() > 0, "Sales should be positive"
    
    # Check categorical values
    assert set(df["region"].unique()).issubset(set(REGIONS)), "Invalid regions"
    assert set(df["product"].unique()).issubset(set(PRODUCTS.keys())), "Invalid products"
    
    # Check calculated fields
    calculated_sales = df["price"] * df["quantity"]
    sales_diff = (calculated_sales - df["sales"]).abs().max()
    assert sales_diff < 0.01, f"Sales calculation error: max diff = {sales_diff}"
    
    logger.info("✅ Dataset validation passed")
    return True


def print_dataset_summary(df: pd.DataFrame):
    """
    Print summary statistics for the dataset.
    
    Args:
        df: Generated DataFrame
    """
    logger.info("\n" + "="*60)
    logger.info("DATASET SUMMARY")
    logger.info("="*60)
    
    logger.info(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    logger.info(f"\nDate Range:")
    logger.info(f"  Start: {df['date'].min().date()}")
    logger.info(f"  End:   {df['date'].max().date()}")
    logger.info(f"  Days:  {(df['date'].max() - df['date'].min()).days}")
    
    logger.info(f"\nRegions:")
    for region, count in df["region"].value_counts().items():
        logger.info(f"  {region:10s}: {count:6,} ({count/len(df)*100:5.2f}%)")
    
    logger.info(f"\nProducts:")
    for product, count in df["product"].value_counts().items():
        logger.info(f"  {product:15s}: {count:6,} ({count/len(df)*100:5.2f}%)")
    
    logger.info(f"\nNumerical Statistics:")
    logger.info(f"  Price:    ${df['price'].mean():,.2f} ± ${df['price'].std():,.2f} (range: ${df['price'].min():.2f} - ${df['price'].max():.2f})")
    logger.info(f"  Quantity: {df['quantity'].mean():.1f} ± {df['quantity'].std():.1f} (range: {df['quantity'].min()} - {df['quantity'].max()})")
    logger.info(f"  Sales:    ${df['sales'].mean():,.2f} ± ${df['sales'].std():,.2f} (range: ${df['sales'].min():.2f} - ${df['sales'].max():.2f})")
    
    logger.info(f"\nTotal Revenue: ${df['sales'].sum():,.2f}")
    logger.info(f"Avg Daily Revenue: ${df.groupby('date')['sales'].sum().mean():,.2f}")
    
    logger.info(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    logger.info("="*60 + "\n")


def save_dataset(df: pd.DataFrame, output_path: str, file_format: str = "csv"):
    """
    Save dataset to file.
    
    Args:
        df: DataFrame to save
        output_path: Output file path
        file_format: Output format ('csv' or 'parquet')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving dataset to: {output_path}")
    
    if file_format == "csv":
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Saved as CSV: {output_path.stat().st_size / 1024**2:.2f} MB")
    
    elif file_format == "parquet":
        if not PARQUET_AVAILABLE:
            logger.error("PyArrow not installed. Install with: pip install pyarrow")
            sys.exit(1)
        
        df.to_parquet(output_path, index=False, compression="snappy")
        logger.info(f"✅ Saved as Parquet: {output_path.stat().st_size / 1024**2:.2f} MB")
    
    else:
        raise ValueError(f"Unsupported format: {file_format}")


# =====================================================
# COMMAND LINE INTERFACE
# =====================================================

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic sales data for MLOps training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10,000 rows
  python data/generate_data.py --rows 10000
  
  # Specify date range
  python data/generate_data.py --start-date 2023-01-01 --end-date 2024-12-31
  
  # Save as Parquet
  python data/generate_data.py --format parquet --output data/raw/sales_data.parquet
  
  # Quiet mode (no progress)
  python data/generate_data.py --rows 50000 --quiet
        """
    )
    
    parser.add_argument(
        "--rows",
        type=int,
        default=10000,
        help="Number of rows to generate (default: 10000)"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date (YYYY-MM-DD) (default: 2023-01-01)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD) (default: 2024-12-31)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/sales_data.csv",
        help="Output file path (default: data/raw/sales_data.csv)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet"],
        default="csv",
        help="Output format (default: csv)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED})"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress updates"
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip dataset validation"
    )
    
    return parser.parse_args()


# =====================================================
# MAIN FUNCTION
# =====================================================

def main():
    """
    Main function to generate sales data.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Set logging level
    if args.quiet:
        logger.setLevel(logging.WARNING)
    
    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        logger.error("Use format: YYYY-MM-DD")
        sys.exit(1)
    
    # Validate dates
    if start_date >= end_date:
        logger.error("Start date must be before end date")
        sys.exit(1)
    
    # Log configuration
    logger.info("="*60)
    logger.info("SALES DATA GENERATOR")
    logger.info("="*60)
    logger.info(f"Configuration:")
    logger.info(f"  Rows:       {args.rows:,}")
    logger.info(f"  Start Date: {start_date.date()}")
    logger.info(f"  End Date:   {end_date.date()}")
    logger.info(f"  Output:     {args.output}")
    logger.info(f"  Format:     {args.format}")
    logger.info(f"  Seed:       {args.seed}")
    logger.info("="*60 + "\n")
    
    try:
        # Generate dataset
        df = generate_dataset(
            num_rows=args.rows,
            start_date=start_date,
            end_date=end_date,
            show_progress=not args.quiet
        )
        
        # Validate dataset
        if not args.no_validate:
            validate_dataset(df)
        
        # Print summary
        if not args.quiet:
            print_dataset_summary(df)
        
        # Save dataset
        save_dataset(df, args.output, args.format)
        
        logger.info(f"\n✅ SUCCESS! Dataset generated and saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}", exc_info=True)
        sys.exit(1)


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
