# %%
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
plt.figure(dpi=150)
import math
import numpy as np

# Configuration
csv_file = r"C:\Users\chh03\SMWE\Ops Data and Supply Analytics - Supply Planning and Analytics\Reporting\VR_Workflow\BaseDataFormat_Input_01.06.2026 CV.csv"
base_path = Path('C:/Users/chh03/SMWE/Ops Data and Supply Analytics - Analytics and Items Team Workflow/Reporting/VR_Workflow/Finished_Data')
folder_name = 'FInal_Sales_Plan_Output_V1'
varietal_subfolder = 'Varietal'
brand_subfolder = 'Brand'

# Read data with Polars
df = pl.read_csv(csv_file)
df = df.fill_null(0)

# Month mapping
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

# Helper functions
def create_year_string(col: str) -> str:
    """Extract year from column name"""
    return col.split("_")[-1]

def extract_first_year(col_list: list) -> str:
    """Extract year from first column in list"""
    return col_list[0].split("_")[-1] if col_list else None

def extract_last_year(col_list: list) -> str:
    """Extract year from last column in list"""
    return col_list[-1].split("_")[-1] if col_list else None

# Add Month_add column
df = df.with_columns(
    (pl.col('m_bottle_prep') + pl.col('m_pre_bottle_prep')).alias('Month_add')
)

# Get column lists
inv_plan_cols = [col for col in df.columns if col.startswith("Inventory_")]
inv_first_col_year = extract_first_year(inv_plan_cols)
inv_last_col_year = extract_last_year(inv_plan_cols)

sales_plan_cols = [col for col in df.columns if col.startswith("Sales_Plan_")]
sales_first_col_year = extract_first_year(sales_plan_cols)
sales_last_col_year = extract_last_year(sales_plan_cols)

# Convert numeric columns
numeric_cols = [col for col in df.columns if col.startswith("Sales_Plan_") or col.startswith("Backup_Plan_")]
df = df.with_columns([
    pl.col(col).cast(pl.Float64, strict=False) for col in numeric_cols + inv_plan_cols
])

# Create Current_date column
year_var = 2026
month_var = 1
day_var = 1
input_date = datetime(year=year_var, month=month_var, day=day_var).date()
current_date = input_date.strftime("%Y-%m-%d")

df = df.with_columns(
    pl.lit(current_date).str.strptime(pl.Date, "%Y-%m-%d").alias('Current_date')
)

print("DataFrame loaded. Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# %%
# Main inventory runout calculation loop
# This section calculates how many months of inventory are available for each year

print("\n=== STARTING MAIN INVENTORY RUNOUT CALCULATION ===")

# Initialize runout columns for each inventory year
for col in inv_plan_cols:
    year = create_year_string(col)
    df = df.with_columns(pl.lit(0.0).alias(f'Runout_Months_{year}'))

# Convert to pandas temporarily for the complex iterative logic
# Note: This section is performance-critical and uses complex state tracking
# that's challenging to vectorize in Polars
df_pd = df.to_pandas()

for idx in range(len(df_pd)):
    row = df_pd.iloc[idx]
    
    if row['Status'] != "Active":
        continue
    
    # Initialize state variables for tracking inventory allocation
    indexer = 0  # Controls skipping already-utilized inventory columns
    inventory_carryover = 0  # Inventory carried forward from previous year
    sales_carryover = 0  # Unmet sales carried forward to next inventory
    months_stored = 0  # Accumulated months of coverage
    total_sales = 0
    total_inventory = 0
    
    for sal_col in sales_plan_cols:
        current_year_dt = datetime(year_var, month_var, day_var)
        sale_year = int(create_year_string(sal_col))
        
        # Adjust first year sales for partial year
        if year_var == sale_year:
            balance_of_months = 12 - current_year_dt.month + 1
            remaining_year_proportion = balance_of_months / 12
            sale_value = row[sal_col] * remaining_year_proportion
            monthly_sale_value = row[sal_col] / 12
            total_sales += row[sal_col]
        else:
            sale_value = row[sal_col]
            monthly_sale_value = sale_value / 12
            total_sales += sale_value
        
        # Skip if no sales planned
        if sale_value == 0:
            break
        
        # Iterate through inventory columns starting from current indexer
        for col in inv_plan_cols[indexer:]:
            year = create_year_string(col)
            inv_value = row[col]
            total_inventory += inv_value
            
            # CASE 1: Zero inventory
            if inv_value == 0:
                df_pd.loc[idx, f'Runout_Months_{year}'] = 0
                indexer += 1
                continue
            
            # CASE 2: No carryover from previous iterations
            elif (inventory_carryover == 0) and (sales_carryover == 0):
                take_value = min(sale_value, inv_value)
                sales_difference = sale_value - take_value
                inventory_difference = inv_value - take_value
                
                # SUBCASE 2a: Exact match - inventory equals sales
                if (sales_difference == 0) and (inventory_difference == 0):
                    months_stored = inv_value / monthly_sale_value
                    df_pd.loc[idx, f'Runout_Months_{year}'] = months_stored
                    # Reset all carryover variables
                    inventory_carryover = 0
                    sales_carryover = 0
                    months_stored = 0
                    indexer += 1
                    break
                
                # SUBCASE 2b: More inventory than sales
                elif sales_difference == 0:
                    months_stored += sale_value / monthly_sale_value
                    inventory_carryover = inventory_difference
                    break
                
                # SUBCASE 2c: More sales than inventory
                elif inventory_difference == 0:
                    months_stored = int(inv_value / monthly_sale_value)
                    df_pd.loc[idx, f'Runout_Months_{year}'] = months_stored
                    sales_carryover = sales_difference
                    months_stored = 0
                    indexer += 1
            
            # CASE 3: Sales carryover but no inventory carryover
            elif (inventory_carryover == 0) and (sales_carryover != 0):
                take_value = min(sales_carryover, inv_value)
                sales_difference = sales_carryover - take_value
                inventory_difference = inv_value - take_value
                
                # SUBCASE 3a: Inventory covers all carryover sales
                if sales_difference == 0:
                    months_stored = sales_carryover / monthly_sale_value
                    inventory_carryover = inventory_difference
                    sales_carryover = 0
                    break
                
                # SUBCASE 3b: Inventory exhausted, sales carryover remains
                elif inventory_difference == 0:
                    result = int(inv_value / monthly_sale_value)
                    df_pd.loc[idx, f'Runout_Months_{year}'] = result
                    sales_carryover = sales_difference
                    indexer += 1
            
            # CASE 4: Inventory carryover but no sales carryover
            elif (inventory_carryover != 0) and (sales_carryover == 0):
                take_value = min(sale_value, inventory_carryover)
                sales_difference = sale_value - take_value
                inventory_difference = inventory_carryover - take_value
                
                # SUBCASE 4a: More inventory than sales
                if sales_difference == 0:
                    months_stored += sale_value / monthly_sale_value
                    inventory_carryover = inventory_difference
                    break
                
                # SUBCASE 4b: Carryover inventory exhausted
                elif inventory_difference == 0:
                    result = int(months_stored + (inventory_carryover / monthly_sale_value))
                    df_pd.loc[idx, f'Runout_Months_{year}'] = result
                    inventory_carryover = 0
                    months_stored = 0
                    sales_carryover = sales_difference
                    indexer += 1

# Convert back to Polars
df = pl.from_pandas(df_pd)
df = df.fill_null(0)

print("\n=== RUNOUT CALCULATION COMPLETE ===")
print(f"Processed {len(df)} rows")

# %%
# Generate runout dates and ideal release/runout dates
print("\n=== GENERATING RUNOUT DATE COLUMNS ===")

for col in inv_plan_cols:
    year = create_year_string(col)
    year_start = int(year) + 1
    start_year_string = f"{year_start}-01-01"
    
    # Calculate ideal release date
    # Formula: Start of next year + prep months
    df = df.with_columns([
        (pl.col('Month_add').cast(pl.Int64)).alias('Month_add_int')
    ])
    
    # Create ideal release and runout dates
    df = df.with_columns([
        (pl.lit(start_year_string).str.strptime(pl.Date, "%Y-%m-%d")
         .dt.offset_by(pl.col('Month_add_int').cast(pl.String) + "mo"))
        .alias(f'Ideal_Release_{year}'),
    ])
    
    df = df.with_columns([
        pl.col(f'Ideal_Release_{year}')
        .dt.offset_by("12mo")
        .alias(f'Ideal_Runout_{year}')
    ])
    
    # Calculate actual runout dates
    if year == inv_first_col_year:
        # First year: Current date + runout months
        df = df.with_columns([
            (pl.col('Current_date')
             .dt.offset_by(pl.col(f'Runout_Months_{year}').cast(pl.Int64).cast(pl.String) + "mo"))
            .alias(f'Actual_Runout_{year}')
        ])
    elif year <= inv_last_col_year:
        # Subsequent years: Previous actual runout + runout months
        prev_year = str(int(year) - 1)
        df = df.with_columns([
            (pl.col(f'Actual_Runout_{prev_year}')
             .dt.offset_by(pl.col(f'Runout_Months_{year}').cast(pl.Int64).cast(pl.String) + "mo"))
            .alias(f'Actual_Runout_{year}')
        ])
    
    # Calculate Off_Ideal (difference in months)
    df = df.with_columns([
        ((pl.col(f'Actual_Runout_{year}').cast(pl.Int64) - 
          pl.col(f'Ideal_Runout_{year}').cast(pl.Int64)) / 
         (1000000 * 60 * 60 * 24 * 30)).cast(pl.Int64)
        .alias(f'Off_Ideal_{year}')
    ])

print("Runout date columns generated")

# %%
# Generate future bulk need columns
print("\n=== GENERATING FUTURE BULK NEED COLUMNS ===")

# Convert to pandas for complex conditional logic
df_pd = df.to_pandas()

off_ideal_cols = [col for col in df_pd.columns if col.startswith("Off_Ideal_")]
actual_runout_cols = [col for col in df_pd.columns if col.startswith("Actual_Runout_")]

for idx in range(len(df_pd)):
    row = df_pd.iloc[idx]
    starting_year = inv_last_col_year
    market_off = row['market_off_ideal']
    status_info = row['Status']
    
    runout_last_year = actual_runout_cols[-1]
    runout_months_last = off_ideal_cols[-1]
    runout_date_value = row[runout_last_year]
    runout_months_value = row[runout_months_last]
    
    product_plan_value = row['Product_Runout_Plan']
    runout_months_value_minus_market_max = runout_months_value
    
    create_value_runout = runout_months_value_minus_market_max / 12
    create_mod_math = math.modf(create_value_runout)  # (decimal, integer)
    
    # Determine skip years based on product plan
    if product_plan_value == "Off_Ideal":
        create_mod_skip_year = 3
    else:
        create_mod_skip_year = create_mod_math[1]
    
    create_mod_partial_year = create_mod_math[0]
    partial_production = 1 - create_mod_partial_year
    
    x = 1
    
    # Handle pandas datetime conversion
    if isinstance(runout_date_value, (datetime, pl.Date)):
        runout_year = runout_date_value.year if hasattr(runout_date_value, 'year') else int(str(runout_date_value)[:4])
    else:
        runout_year = int(starting_year)
    
    sales_index_start = sales_plan_cols.index(f'Sales_Plan_{runout_year}') if f'Sales_Plan_{runout_year}' in sales_plan_cols else 0
    
    for sales_col in sales_plan_cols[sales_index_start:]:
        sales_val = row[sales_col]
        
        if product_plan_value == "Cascade":
            if status_info in ["Discontinued", "DTC"]:
                for i in range(0, 7)[x:]:
                    year_offset = int(starting_year) + i
                    df_pd.loc[idx, f'Inventory_Target_{year_offset}'] = 0
                    df_pd.loc[idx, f'Future_Sale_Base_{year_offset}'] = 0
                    df_pd.loc[idx, f'Future_Bulk_{year_offset}'] = 0
            else:
                for i in range(0, 7)[x:]:
                    year_offset = int(starting_year) + i
                    
                    if create_mod_skip_year > 0:
                        # Full year production (50% of sales)
                        result = sales_val * 0.5
                        df_pd.loc[idx, f'Inventory_Target_{year_offset}'] = int((sales_val * 0.5) + (sales_val * 0.16))
                        df_pd.loc[idx, f'Future_Sale_Base_{year_offset}'] = int(sales_val)
                        df_pd.loc[idx, f'Future_Bulk_{year_offset}'] = int(result)
                        create_mod_skip_year -= 1
                        x += 1
                    
                    elif (create_mod_skip_year == 0) and (create_mod_partial_year != 0):
                        # Partial year production
                        result = sales_val * partial_production
                        four_month_check = sales_val * 0.3333
                        
                        if result < four_month_check:
                            df_pd.loc[idx, f'Inventory_Target_{year_offset}'] = int((sales_val * 0.5) + (sales_val * 0.16))
                            df_pd.loc[idx, f'Future_Sale_Base_{year_offset}'] = int(sales_val)
                            df_pd.loc[idx, f'Future_Bulk_{year_offset}'] = int(four_month_check)
                        else:
                            df_pd.loc[idx, f'Inventory_Target_{year_offset}'] = int((sales_val * 0.5) + (sales_val * 0.16))
                            df_pd.loc[idx, f'Future_Sale_Base_{year_offset}'] = int(sales_val)
                            df_pd.loc[idx, f'Future_Bulk_{year_offset}'] = int(result)
                        
                        create_mod_partial_year = 0
                        x += 1
                        break
                    
                    else:
                        # Full year production after partial year
                        result = sales_val
                        df_pd.loc[idx, f'Inventory_Target_{year_offset}'] = int((sales_val * 0.5) + (sales_val * 0.16))
                        df_pd.loc[idx, f'Future_Sale_Base_{year_offset}'] = int(sales_val)
                        df_pd.loc[idx, f'Future_Bulk_{year_offset}'] = int(result)
                        x += 1
                        break
        
        elif product_plan_value == "Off_Ideal":
            for i in range(0, 7)[x:]:
                year_offset = int(starting_year) + i
                
                if (create_mod_skip_year > 0) and (runout_months_value >= market_off):
                    # Add percentage based on how many years off-ideal
                    percent_add_on = (create_mod_skip_year * 0.25) * sales_val
                    result = sales_val + percent_add_on
                    df_pd.loc[idx, f'Inventory_Target_{year_offset}'] = int((sales_val * 0.5) + (sales_val * 0.16))
                    df_pd.loc[idx, f'Future_Sale_Base_{year_offset}'] = int(sales_val)
                    df_pd.loc[idx, f'Future_Bulk_{year_offset}'] = int(result)
                    create_mod_skip_year -= 1
                    x += 1
                else:
                    # Back to normal production
                    result = sales_val
                    df_pd.loc[idx, f'Inventory_Target_{year_offset}'] = int((sales_val * 0.5) + (sales_val * 0.16))
                    df_pd.loc[idx, f'Future_Sale_Base_{year_offset}'] = int(sales_val)
                    df_pd.loc[idx, f'Future_Bulk_{year_offset}'] = int(result)
                    x += 1
                    break

# Convert back to Polars
df = pl.from_pandas(df_pd)
df = df.fill_null(0)

print("Future bulk need columns generated")

# %%
# Calculate runout, release dates, and graph volumes for future years
def calculate_runout_release_and_graph(df: pl.DataFrame, current_year: int) -> pl.DataFrame:
    """
    Calculate runout months, actual runout dates, ideal release dates, and graph volumes
    for future bulk production years.
    """
    future_bulk_cols = [col for col in df.columns if col.startswith('Future_Bulk_')]
    future_years = sorted([int(col.split('_')[-1]) for col in future_bulk_cols])
    
    # Convert to pandas for complex date operations
    df_pd = df.to_pandas()
    
    for year in future_years:
        bulk_col = f'Future_Bulk_{year}'
        sale_col = f'Future_Sale_Base_{year}'
        runout_col = f'Runout_Months_{year}'
        off_ideal_col = f'Off_Ideal_{year}'
        actual_prev_col = f'Actual_Runout_{year-1}'
        actual_col = f'Actual_Runout_{year}'
        ideal_prev_col = f'Ideal_Release_{year-1}'
        ideal_col = f'Ideal_Release_{year}'
        graph_col = f'Graph_Volume_{year}'
        actual_inventory_col = f'Actual_Inventory_{year}'
        
        if all(col in df_pd.columns for col in [bulk_col, sale_col, actual_prev_col, ideal_prev_col]):
            bulk = df_pd[bulk_col].fillna(0)
            sale = df_pd[sale_col].fillna(0)
            
            # Calculate planned months (as float for precision)
            planned_months_float = np.where(sale > 0, bulk * 12.0 / sale, 0.0)
            
            # Round UP to integer months (ceiling)
            months_int = np.where(sale > 0, np.ceil(planned_months_float), 0).astype(np.int64)
            
            # Persist runout months and Off_Ideal
            df_pd[runout_col] = months_int
            df_pd[off_ideal_col] = 0
            
            # Actual runout date: add integer months to previous actual date
            df_pd[actual_col] = df_pd.apply(
                lambda row: (row[actual_prev_col] + relativedelta(months=int(row[runout_col])))
                if pd.notnull(row[actual_prev_col]) and pd.notnull(row[runout_col])
                else pd.NaT,
                axis=1
            )
            
            # Ideal release date: +12 months to previous ideal date
            df_pd[ideal_col] = df_pd[ideal_prev_col].apply(
                lambda d: (d + relativedelta(months=12)) if pd.notnull(d) else pd.NaT
            )
            
            # Graph volume
            if year <= current_year and actual_inventory_col in df_pd.columns:
                df_pd[graph_col] = df_pd[actual_inventory_col]
            else:
                df_pd[graph_col] = bulk
    
    return pl.from_pandas(df_pd)

df = calculate_runout_release_and_graph(df, current_year=2025)
print("Runout, release, and graph calculations complete")

# %%
# Generate barrel information and Non-Vintage (NV) calculations
print("\n=== GENERATING BARREL AND NV CALCULATIONS ===")

future_bulk_cols = [col for col in df.columns if col.startswith("Future_Bulk_")]

# Replace any remaining nulls with 0 for calculations
df = df.fill_null(0)

for future_col in future_bulk_cols:
    year = create_year_string(future_col)
    barrel_multiplier = 23
    
    # Calculate NV and bulk percentages
    df = df.with_columns([
        (pl.col(future_col) * (pl.col('nv_perc') / 100)).alias(f'NV_Use_Up_{year}'),
        (pl.col(future_col) * (pl.col('bulk_percentage') / 100)).alias(f'Bulk_Need_{year}'),
    ])
    
    # Calculate grape need (total - NV)
    df = df.with_columns([
        (pl.col(future_col) - pl.col(f'NV_Use_Up_{year}')).alias(f'Grape_Need_{year}')
    ])
    
    # Barrel calculations
    df = df.with_columns([
        # Total barrel needs
        (pl.col(f'Grape_Need_{year}') * (pl.col('barrel_percentage') / 100)).alias(f'Total_Barrel_FCE_Need_{year}'),
        (pl.col(f'Grape_Need_{year}') * (pl.col('barrel_percentage') / 100) / barrel_multiplier).alias(f'Total_Barrel_BBL_Need_{year}'),
        
        # Barrel type breakdowns
        (pl.col(f'Grape_Need_{year}') * (pl.col('fo_percentage') / 100) / barrel_multiplier).alias(f'FO_BBL_Need_{year}'),
        (pl.col(f'Grape_Need_{year}') * (pl.col('ao_percentage') / 100) / barrel_multiplier).alias(f'AO_BBL_Need_{year}'),
        (pl.col(f'Grape_Need_{year}') * (pl.col('neutral_percentage') / 100) / barrel_multiplier).alias(f'Neutral_BBL_Need_{year}'),
        
        # Insert barrel calculations
        (pl.col(f'Grape_Need_{year}') * (pl.col('insert_percentage') / 100)).alias(f'Insert_Barrel_FCE_Need_{year}'),
        (pl.col(f'Grape_Need_{year}') * (pl.col('insert_percentage') / 100) / barrel_multiplier).alias(f'Insert_Barrel_BBL_Need_{year}'),
        
        # Stainless and adjunct calculations
        (pl.col(f'Grape_Need_{year}') * (pl.col('stainless_percentage') / 100)).alias(f'Stainless_FCE_Need_{year}'),
        (pl.col(f'Grape_Need_{year}') * (pl.col('adjunct_percentage') / 100)).alias(f'Adjunct_FCE_Need_{year}'),
        (pl.col(f'Grape_Need_{year}') * (pl.col('stainlessBlock_percentage') / 100)).alias(f'Block_FCE_Need_{year}'),
    ])
    
    # Cast to integers for final output
    barrel_cols = [
        f'NV_Use_Up_{year}', f'Bulk_Need_{year}', f'Grape_Need_{year}',
        f'Total_Barrel_FCE_Need_{year}', f'Total_Barrel_BBL_Need_{year}',
        f'FO_BBL_Need_{year}', f'AO_BBL_Need_{year}', f'Neutral_BBL_Need_{year}',
        f'Insert_Barrel_FCE_Need_{year}', f'Insert_Barrel_BBL_Need_{year}',
        f'Stainless_FCE_Need_{year}', f'Adjunct_FCE_Need_{year}', f'Block_FCE_Need_{year}'
    ]
    
    df = df.with_columns([
        pl.col(col).cast(pl.Int64) for col in barrel_cols if col in df.columns
    ])

print("Barrel and NV calculations complete")
print("\nFinal DataFrame shape:", df.shape)

# %%
# CHART GENERATION SECTION
# Helper functions for chart generation

def sanitize_filename(name: str) -> str:
    """Clean filename for safe file system usage"""
    return "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in str(name)).strip().replace(" ", "_")

def ensure_left_margin(ax, pad_pts=10, min_left=0.06, max_left=0.35):
    """Dynamically adjust figure left margin for y-tick labels"""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    
    widths_px = [lbl.get_window_extent(renderer=renderer).width
                 for lbl in ax.get_yticklabels()
                 if lbl.get_visible()]
    
    if not widths_px:
        return
    
    max_w_in = max(widths_px) / fig.dpi
    pad_in = pad_pts / 72.0
    fig_w_in = fig.get_size_inches()[0]
    
    required_left = (max_w_in + pad_in) / fig_w_in
    required_left = max(min(required_left, max_left), min_left)
    plt.subplots_adjust(left=required_left)

def plot_runout_chart_for_subset(
    df_sub: pl.DataFrame,
    outfile: str,
    title: str,
    max_year: int = 2026,
    ideal_mode: str = "future",
    current_year: int = 2025,
    dpi: int = 300,
    filter_status_active: bool = False,
):
    """
    Build and save a landscape runout chart for a given dataframe subset.
    Returns True if chart was saved, False if subset had no usable data.
    """
    if df_sub.shape[0] == 0:
        return False
    
    # Convert to pandas for matplotlib compatibility
    df_pd = df_sub.to_pandas()
    
    # Optional status filter
    if filter_status_active and "Status" in df_pd.columns:
        df_pd = df_pd[df_pd["Status"] == "Active"].copy()
        if df_pd.empty:
            return False
    
    # Convert date columns
    date_cols = [c for c in df_pd.columns if c.startswith(("Actual_Runout_", "Ideal_Release_"))] + ["Current_date"]
    date_cols = [c for c in date_cols if c in df_pd.columns]
    if date_cols:
        df_pd[date_cols] = df_pd[date_cols].apply(pd.to_datetime, errors="coerce")
    
    # Detect vintages and cap to max_year
    years_all = sorted({int(c.split("_")[-1]) for c in df_pd.columns if c.startswith("Actual_Runout_")})
    years = [y for y in years_all if y <= max_year]
    if not years:
        return False
    
    # Axis extent
    min_date_all = df_pd["Current_date"].min() if "Current_date" in df_pd.columns else None
    max_date_series = df_pd[[f"Actual_Runout_{y}" for y in years]].max(axis=1) if years else pd.Series(dtype="datetime64[ns]")
    max_date_data = max_date_series.max() if not max_date_series.empty else None
    max_date_cap = datetime(max_year, 12, 31)
    xmax_date = min(max_date_data, max_date_cap) if pd.notnull(max_date_data) else max_date_cap
    
    # Plot settings
    ann_fontsize = 8
    ann_color = "white"
    ann_ha = "center"
    ann_va = "center"
    cmap = plt.get_cmap("tab10")
    color_map = {year: cmap(i % cmap.N) for i, year in enumerate(years)}
    
    # Landscape US Letter dimensions
    dynamic_height = min(len(df_pd) * 0.6 + 1, 8.5)
    fig, ax = plt.subplots(figsize=(11, dynamic_height), constrained_layout=False)
    
    # Draw bars for each row
    for i, row in enumerate(df_pd.itertuples()):
        drawn_years = set()
        current = getattr(row, "Current_date", None)
        
        # Find first segment start
        start_index = None
        for idx, y in enumerate(years):
            run = getattr(row, f"Actual_Runout_{y}", None)
            if pd.notnull(run) and (current is None or current < run):
                start_index = idx
                break
        
        if start_index is not None:
            # First segment
            first = years[start_index]
            finish = getattr(row, f"Actual_Runout_{first}", None)
            graph_val = getattr(row, f"Graph_Volume_{first}", 0)
            
            if pd.notnull(finish):
                finish_clipped = min(finish, max_date_cap)
                if current is None or current < finish_clipped:
                    start_num = mdates.date2num(current) if pd.notnull(current) else mdates.date2num(finish_clipped)
                    finish_num = mdates.date2num(finish_clipped)
                    width = max(finish_num - start_num, 0)
                    
                    if width > 0:
                        ax.barh(i, width, left=start_num, height=0.5, color=color_map[first], edgecolor="grey")
                        drawn_years.add(first)
                        
                        # Annotation
                        if first <= current_year:
                            text = f"RO:{getattr(row, f'Runout_Months_{first}', '')}, OI:{getattr(row, f'Off_Ideal_{first}', '')}"
                        else:
                            text = f"Vol:{int(graph_val/1000)}K FCE"
                        ax.text(start_num + width / 2, i, text, color=ann_color, fontsize=ann_fontsize, ha=ann_ha, va=ann_va)
            
            # Subsequent segments
            for j in range(start_index, len(years) - 1):
                y, ny = years[j], years[j + 1]
                s = getattr(row, f"Actual_Runout_{y}", None)
                fdt = getattr(row, f"Actual_Runout_{ny}", None)
                graph_val_next = getattr(row, f"Graph_Volume_{ny}", 0)
                
                if pd.isnull(s) or pd.isnull(fdt):
                    continue
                    
                s_clipped = min(s, max_date_cap)
                fdt_clipped = min(fdt, max_date_cap)
                
                if s_clipped >= fdt_clipped:
                    continue
                
                s_num = mdates.date2num(s_clipped)
                fdt_num = mdates.date2num(fdt_clipped)
                width_next = fdt_num - s_num
                
                ax.barh(i, width_next, left=s_num, height=0.5, color=color_map[ny], edgecolor="grey")
                drawn_years.add(ny)
                
                if ny <= current_year:
                    text = f"RO:{getattr(row, f'Runout_Months_{ny}', '')}, OI:{getattr(row, f'Off_Ideal_{ny}', '')}"
                else:
                    text = f"Vol:{int(graph_val_next/1000)}K FCE"
                ax.text(s_num + width_next / 2, i, text, color=ann_color, fontsize=ann_fontsize, ha=ann_ha, va=ann_va)
        
        # Ideal markers
        for y in drawn_years:
            ideal = getattr(row, f"Ideal_Release_{y}", None)
            if pd.notnull(ideal):
                if ideal_mode == "future" and current is not None and ideal <= current:
                    continue
                ideal_num = mdates.date2num(min(ideal, max_date_cap))
                ax.vlines(ideal_num, i - 0.2, i + 0.2, linestyles="dotted", colors="black")
                label = f"(V{y} - {mdates.num2date(ideal_num).strftime('%b %Y')})"
                ax.text(ideal_num, i + 0.25, label, color="black", ha="center", va="bottom", fontsize=8)
    
    # Configure axes
    min_num = mdates.date2num(min_date_all) if pd.notnull(min_date_all) else mdates.date2num(datetime(current_year, 1, 1))
    max_num = mdates.date2num(xmax_date)
    ax.set_xlim(min_num, max_num)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlabel("Calendar Date", fontsize=12, labelpad=6)
    plt.xticks(rotation=30)
    
    ax.set_yticks(range(len(df_pd)))
    
    # Build y-labels with Brand info
    if "Brand" in df_pd.columns:
        desc = df_pd["description"].fillna("").astype(str)
        brand = df_pd["Brand"].fillna("").astype(str)
        y_labels = [f"{d}|{b}" if b.strip() else d for d, b in zip(desc, brand)]
    else:
        y_labels = (
            df_pd["description"].astype(str)
            if "description" in df_pd.columns
            else df_pd.index.astype(str).tolist()
        )
    
    ax.set_yticklabels(y_labels)
    
    # Legend
    legend_handles = [mpatches.Patch(color=color_map[y], label=str(y)) for y in years]
    ax.legend(handles=legend_handles, title="Vintage", bbox_to_anchor=(1.02, 1), 
              loc="upper left", borderaxespad=0.0, frameon=False)
    
    ax.margins(x=0.01, y=0.02)
    ensure_left_margin(ax, pad_pts=10, min_left=0.06, max_left=0.35)
    
    plt.title(title)
    
    outfile_path = Path(outfile)
    outfile_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(outfile_path), dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return True

def save_charts_by_brand_and_varietal(
    df: pl.DataFrame,
    base_path: Path,
    folder_name: str,
    varietal_subfolder: str = "Varietal",
    brand_subfolder: str = "Brand",
    max_year: int = 2028,
    ideal_mode: str = "future",
    current_year: int = 2025,
    dpi: int = 300,
    filter_status_active: bool = False,
):
    """
    Generate PNG charts for each Varietal and Brand
    """
    scenario_path = base_path / folder_name
    varietal_dir = scenario_path / varietal_subfolder
    brand_dir = scenario_path / brand_subfolder
    varietal_dir.mkdir(parents=True, exist_ok=True)
    brand_dir.mkdir(parents=True, exist_ok=True)
    
    saved_varietal, saved_brand = 0, 0
    
    # Per Varietal
    if "Varietal" in df.columns:
        varietals = sorted(df.select("Varietal").unique().drop_nulls().to_series().to_list())
        for var in varietals:
            subset = df.filter(pl.col("Varietal") == var)
            outfile = varietal_dir / f"{sanitize_filename(var)}.png"
            title = f"Runout – Varietal: {var} (≤ {max_year})"
            ok = plot_runout_chart_for_subset(
                subset, str(outfile), title, max_year, ideal_mode, current_year, dpi, filter_status_active
            )
            if ok:
                saved_varietal += 1
    
    # Per Brand
    if "Brand" in df.columns:
        brands = sorted(df.select("Brand").unique().drop_nulls().to_series().to_list())
        for brand in brands:
            subset = df.filter(pl.col("Brand") == brand)
            outfile = brand_dir / f"{sanitize_filename(brand)}.png"
            title = f"Runout – Brand: {brand} (≤ {max_year})"
            ok = plot_runout_chart_for_subset(
                subset, str(outfile), title, max_year, ideal_mode, current_year, dpi, filter_status_active
            )
            if ok:
                saved_brand += 1
    
    return {
        "output_root": str((base_path / folder_name).resolve()),
        "varietal_folder": str(varietal_dir.resolve()),
        "brand_folder": str(brand_dir.resolve()),
        "varietal_charts_saved": saved_varietal,
        "brand_charts_saved": saved_brand,
    }

# Generate charts
print("\n=== GENERATING CHARTS ===")
summary = save_charts_by_brand_and_varietal(
    df=df,
    base_path=base_path,
    folder_name=folder_name,
    varietal_subfolder=varietal_subfolder,
    brand_subfolder=brand_subfolder,
    max_year=2028,
    ideal_mode="future",
    current_year=2025,
    dpi=300,
    filter_status_active=True
)
print("Charts generated:", summary)

# %%
# REPORTING SECTION

# Top 50 Export
print("\n=== GENERATING TOP 50 REPORT ===")
off_ideal_cols = [col for col in df.columns if col.startswith("Off_Ideal_")]
if off_ideal_cols and sales_plan_cols:
    select_cols = ['Brand', 'Concat', 'description', 'Varietal', 'Roll_Up', 'Color', off_ideal_cols[-1], sales_plan_cols[0]]
    df_top_50 = df.select([col for col in select_cols if col in df.columns])
    df_top_50_sorted = df_top_50.sort(sales_plan_cols[0], descending=True).head(50)
    print(df_top_50_sorted)

# %%
# Inventory by Year grouped by Color and Roll_Up
print("\n=== GENERATING INVENTORY BY YEAR REPORT ===")
if 'Color' in df.columns and 'Roll_Up' in df.columns:
    df_inventory_group = df.group_by(['Color', 'Roll_Up']).agg([
        pl.col(col).sum() for col in inv_plan_cols
    ])
    print(df_inventory_group)

# %%
# Sales Report - Cases and Tons
print("\n=== GENERATING SALES REPORTS ===")

# Create base report DataFrame
report_cols = ['Concat', 'description', 'Roll_Up', 'Color'] + sales_plan_cols
df_fiscal_total_cases = df.select([col for col in report_cols if col in df.columns])

# Group by Roll_Up and Color
df_report_sales_cases = df_fiscal_total_cases.group_by('Roll_Up').agg([
    pl.col(col).sum() for col in sales_plan_cols
])

df_report_sales_color = df_fiscal_total_cases.group_by('Color').agg([
    pl.col(col).sum() for col in sales_plan_cols
])

print("\nSales by Color (Cases):")
print(df_report_sales_color)

# Convert to tons
# Note: RSL uses 60 cases/ton, others use 65 cases/ton
df_pd = df_fiscal_total_cases.to_pandas()

def adjust_sales_to_tons(row):
    """Convert cases to tons based on Roll_Up"""
    rollup_value = row['Roll_Up']
    divisor = 60 if rollup_value == 'RSL' else 65
    for col in sales_plan_cols:
        if col in row.index:
            row[col] = row[col] / divisor
    return row

df_fiscal_total_tons = df_pd.apply(adjust_sales_to_tons, axis=1)
df_fiscal_total_tons = pl.from_pandas(df_fiscal_total_tons)

df_report_sales_tons = df_fiscal_total_tons.group_by('Roll_Up').agg([
    pl.col(col).sum() for col in sales_plan_cols
])

df_report_sales_color_tons = df_fiscal_total_tons.group_by(['Color', 'Roll_Up']).agg([
    pl.col(col).sum() for col in sales_plan_cols
])

print("\nSales by Color and Roll_Up (Tons):")
print(df_report_sales_color_tons)

# %%
# Grape Need Report - Convert FCE to Tons
print("\n=== GENERATING GRAPE NEED REPORT (TONS) ===")

grape_fce_cols = [col for col in df.columns if col.startswith(("Future_Bulk_", "NV_Use_Up_", "Grape_Need_"))]
report_base_cols = ['Brand', 'Concat', 'description', 'Varietal', 'Roll_Up', 'Color']
df_report = df.select([col for col in report_base_cols + grape_fce_cols if col in df.columns])

# Convert to pandas for the adjustment function
df_report_pd = df_report.to_pandas()

def adjust_grape_to_tons(row):
    """Convert grape FCE to tons based on Roll_Up"""
    rollup_value = row['Roll_Up']
    divisor = 60 if rollup_value == 'RSL' else 65
    for col in grape_fce_cols:
        if col in row.index:
            row[col] = row[col] / divisor
    return row

df_report_tons_base = df_report_pd.apply(adjust_grape_to_tons, axis=1)
df_report_tons_pl = pl.from_pandas(df_report_tons_base)

df_report_tons = df_report_tons_pl.group_by(['Color', 'Roll_Up']).agg([
    pl.col(col).sum() for col in grape_fce_cols if col in df_report_tons_pl.columns
])

print("\nGrape Need by Color and Roll_Up (Tons):")
print(df_report_tons)

print("\n=== POLARS CONVERSION COMPLETE ===")
print(f"Final DataFrame shape: {df.shape}")
print(f"Total columns: {len(df.columns)}")
