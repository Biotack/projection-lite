# %%
import pandas as pd
from datetime import datetime
from pathlib import Path
from pandas.tseries.offsets import DateOffset
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
plt.figure(dpi=150)
import math as math
from IPython import display
import numpy as np
import plotly as plot

csv_file = r"C:\Users\chh03\SMWE\Ops Data and Supply Analytics - Supply Planning and Analytics\Reporting\VR_Workflow\BaseDataFormat_Input_01.06.2026 CV.csv"
df = pd.read_csv(csv_file)
df.fillna(0, inplace=True)

base_path = Path('C:/Users/chh03/SMWE/Ops Data and Supply Analytics - Analytics and Items Team Workflow/Reporting/VR_Workflow/Finished_Data')
folder_name = f'FInal_Sales_Plan_Output_V1'
varietal_subfolder = f'Varietal'
brand_subfolder = f'Brand'

# df = dfbase[dfbase['Status'] == 'Active'].copy()
pd.set_option('display.max_columns', None)

month_map = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December': 12
}


def create_year_string(col):
    return col.split("_")[-1]

def extract_first_year(list:list):
    result = str.split(list[0], "_")[-1] if list else None
    return result

def extract_last_year(list:list):
    result = str.split(list[-1], "_")[-1] if list else None
    return result

month_add = df['m_bottle_prep'] + df['m_pre_bottle_prep']
df.insert(df.columns.get_loc(f'Concat'), f'Month_add', month_add)


#Handle Inv Cols
inv_plan_cols = [col for col in df.columns if col.startswith("Inventory_")]
inv_first_col_year = extract_first_year(inv_plan_cols)
inv_last_col_year = extract_last_year(inv_plan_cols)
# inv_last_col_year = str(int(inv_last_col_year) + 1)

df[inv_plan_cols] = df[inv_plan_cols].apply(pd.to_numeric, errors="coerce")
numeric_cols = [col for col in df.columns if col.startswith("Sales_Plan_") or col.startswith("Backup_Plan_")]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

#Handle Sales Cols
sales_plan_cols = [col for col in df.columns if col.startswith("Sales_Plan_")]

sales_first_col_year = extract_first_year(sales_plan_cols)
sales_last_col_year = extract_last_year(sales_plan_cols)

# Create Average Sales Column
# average_col = df[sales_plan_cols].mean(axis=1)
# df['Month_add'] = pd.to_numeric(df['Month_add'], errors='coerce')

# if sales_plan_cols:
#     last_sales_plan_idx = df.columns.get_loc(sales_plan_cols[-1])
#     df.insert(last_sales_plan_idx+1, 'Average_Sales', average_col)

def actual_release_calc(col: str, release_col):
    year = create_year_string(col)
    if year == inv_first_col_year:
        return current_date + int(df)
    addition = release_col.apply(lambda x: DateOffset(months=x) + df[f'Inventory_Runout_{year}'])
    result = pd.to_datetime(addition, errors='coerce')
    return result


display.display(df)    

def create_balance_vit_req(year, runout_year):
    for i in range(1,7):
        result = int(year) + i
        return (result, runout_year)


# def make_sales_cols(year):
#     pass

#Create Current Date column

year_var = 2026
month_var = 1
day_var = 1

# Construct the date
input_date = datetime(year=year_var, month=month_var, day=day_var).date()
current_date = input_date.strftime("%Y-%m-%d")

# Insert into DataFrame
df.insert(df.columns.get_loc('Month_add') + 1, 'Current_date', pd.to_datetime(current_date, errors='coerce'))

# display.display(df)


#Main Loop - iterates through each row, and for each row iterates through each sales year, and for each sales year iterates through the inventory years starting at the current sales year.
#Main Loop - handles different statuses and calculates future bulk needs
for row in df.itertuples():
    starting_year = inv_last_col_year
    market_off = getattr(row, 'market_off_ideal')
    status_info = getattr(row, f'Status')
    
    # ===== HANDLE DTC AND Active_NV ITEMS =====
    if status_info in ["DTC", "Active_NV"]:
        # For DTC and Active_NV, simply copy Sales_Plan to Future_Bulk
        for sales_col in sales_plan_cols:
            year = create_year_string(sales_col)
            sales_val = getattr(row, sales_col, 0)
            
            # Only process if the sales value exists and is not zero
            if sales_val > 0:
                df.loc[row.Index, f'Future_Bulk_{year}'] = int(sales_val)
                df.loc[row.Index, f'Future_Sale_Base_{year}'] = int(sales_val)
                df.loc[row.Index, f'Inventory_Target_{year}'] = int((int(sales_val) * 0.5) + (int(sales_val) * 0.16))
            else:
                # If sales plan is 0 or doesn't exist, set to 0
                df.loc[row.Index, f'Future_Bulk_{year}'] = 0
                df.loc[row.Index, f'Future_Sale_Base_{year}'] = 0
                df.loc[row.Index, f'Inventory_Target_{year}'] = 0
        
        # Skip all the complex logic below and move to next row
        continue
    
    # ===== HANDLE DISCONTINUED ITEMS =====
    if status_info == "Discontinued":
        # Set all future columns to 0 for discontinued items
        for sales_col in sales_plan_cols:
            year = create_year_string(sales_col)
            df.loc[row.Index, f'Future_Bulk_{year}'] = 0
            df.loc[row.Index, f'Future_Sale_Base_{year}'] = 0
            df.loc[row.Index, f'Inventory_Target_{year}'] = 0
        continue
    
    # ===== ORIGINAL LOGIC FOR "Active" STATUS CONTINUES BELOW =====
    runout_last_year = actual_runout[-1]
    runout_months_last = off_ideal[-1]
    runout_date_value = getattr(row, runout_last_year)
    #gets market off ideal number
    runout_months_value = getattr(row, runout_months_last)
    print(f'runout_date_value: {runout_date_value}')
    # print(f'runout_months_value: {runout_months_value}')
    # print(f'Item can be: {market_off} months off')
   
    product_plan_value = getattr(row, f'Product_Runout_Plan')
    print(product_plan_value)
    runout_months_value_minus_market_max = runout_months_value
    
    # print(f'runout_months_value_minus_market_max {runout_months_value_minus_market_max}')
    create_value_runout = runout_months_value_minus_market_max/12
    print(f'create_value_runout to split for months of oversupply: {create_value_runout}')
    
    create_mod_math = math.modf(create_value_runout)#decimal as 0,int as 1
    # print(create_mod_math)
    if product_plan_value == "Off_Ideal":
        create_mod_skip_year = 3
    else:
        create_mod_skip_year = create_mod_math[1]
    print(f'create_mod_skip_year: {create_mod_skip_year}, product_plan_value: {product_plan_value}')
    start_create_mod_skip_year = create_mod_math[1]
    create_mod_partial_year = create_mod_math[0]
    partial_production = 1-create_mod_partial_year
    # print(f'partial_production: {partial_production}')
    x = 1
    sales_index_start = sales_plan_cols.index(f'Sales_Plan_{runout_date_value.year}')
    
    for sales_col in sales_plan_cols[sales_index_start:]:
        # print(f'Item: {row.description} Sales Column reference: {sales_col}')
        sales_val = getattr(row, sales_col)
        # print(f'Sales per year: {sales_val}')
        sales_val_per_month = sales_val/12
        # print(f'Sales per Month: {sales_val_per_month}')
        
        if product_plan_value == "Cascade":
            for i in range(0,7)[x:]:
                if create_mod_skip_year != 0:
                    # print(f'Future_Sales_' + str(int(starting_year) + i))
                    # print(sales_val)
                    # print(f'cascade create_mod_skip_year {create_mod_skip_year}')
                    result = sales_val * 0.5
                    df.loc[row.Index, f'Inventory_Target_' + str(int(starting_year) + i)] = int((int(sales_val) * 0.5) + (int(sales_val) * 0.16))
                    df.loc[row.Index, f'Future_Sale_Base_' + str(int(starting_year) + i)] = int(sales_val)
                    df.loc[row.Index, f'Future_Bulk_' + str(int(starting_year) + i)] = int(result)
                    # print(f'cascade Create mod skip year: Setting result to: {result}')
                    result = None
                    create_mod_skip_year -= 1
                    x += 1
                
                elif (create_mod_skip_year == 0) and (create_mod_partial_year != 0):
                    # print(f'Future_Sales_' + str(int(starting_year) + i))
                    # print(f'create_mod_partial {create_mod_partial_year}')
                    result = sales_val * partial_production
                    four_month_check = sales_val * 0.3333
                    if result < four_month_check:
                        df.loc[row.Index, f'Inventory_Target_' + str(int(starting_year) + i)] = int((int(sales_val) * 0.5) + (int(sales_val) * 0.16))
                        df.loc[row.Index, f'Future_Sale_Base_' + str(int(starting_year) + i)] = int(sales_val)
                        df.loc[row.Index, f'Future_Bulk_' + str(int(starting_year) + i)] = four_month_check
                        # print(f'Create mod skip year: Setting result to: Four Months: {four_month_check}')
                    else:
                        df.loc[row.Index, f'Inventory_Target_' + str(int(starting_year) + i)] = int((int(sales_val) * 0.5) + (int(sales_val) * 0.16))
                        df.loc[row.Index, f'Future_Sale_Base_' + str(int(starting_year) + i)] = int(sales_val)
                        df.loc[row.Index, f'Future_Bulk_' + str(int(starting_year) + i)] = int(result)
                        # print(f'Create mod skip year: Setting result to: Sales*Partial: {result}')
                    sales_index_start +=1
                    create_mod_partial_year = 0
                    result = None
                    x +=1
                    
                    break
                else:
                    # (create_mod_partial_year == 0) and (create_mod_skip_year == 0):
                    # print(f'Future_Sales_' + str(int(starting_year) + i))
                    # print(sales_val)
                    result = sales_val
                    df.loc[row.Index, f'Inventory_Target_' + str(int(starting_year) + i)] = int((int(sales_val) * 0.5) + (int(sales_val) * 0.16))
                    df.loc[row.Index, f'Future_Sale_Base_' + str(int(starting_year) + i)] = int(sales_val)
                    df.loc[row.Index, f'Future_Bulk_' + str(int(starting_year) + i)] = int(result)
                    # print(f'Create mod skip year: Setting result to: {result}')
                    result = None
                    sales_index_start += 1
                    x += 1
                    break
        
        if product_plan_value == "Off_Ideal":
            for i in range(0,7)[x:]:
                if (create_mod_skip_year != 0) and (runout_months_value >= market_off):
                    # print(f'Future_Sales_' + str(int(starting_year) + i))
                    # print(sales_val)
                    # print(f'off-ideal_skip_year {create_mod_skip_year}')
                    percent_add_on = (create_mod_skip_year * .25) * sales_val
                    # print(f'percent_add_on: {percent_add_on}')
                    result = sales_val + percent_add_on
                    df.loc[row.Index, f'Inventory_Target_' + str(int(starting_year) + i)] = int((int(sales_val) * 0.5) + (int(sales_val) * 0.16))
                    df.loc[row.Index, f'Future_Sale_Base_' + str(int(starting_year) + i)] = int(sales_val)
                    df.loc[row.Index, f'Future_Bulk_' + str(int(starting_year) + i)] = int(result)
                    # print(f'off-ideal Create mod skip year: Setting result to: {result}')
                    result = None
                    create_mod_skip_year -= 1
                    x += 1
                
                # elif (create_mod_skip_year == 0) and (create_mod_partial_year != 0):
                #     print(f'Future_Sales_' + str(int(starting_year) + i))
                #     print(f'create_mod_partial {create_mod_partial_year}')
                #     result = sales_val * partial_production
                #     four_month_check = sales_val * 0.3333
                #     if result < four_month_check:
                #         df.loc[row.Index, f'Future_Bulk_' + str(int(starting_year) + i)] = four_month_check
                #         print(f'Create mod skip year: Setting result to: Four Months: {four_month_check}')
                #     else:
                #         df.loc[row.Index, f'Future_Bulk_' + str(int(starting_year) + i)] = int(result)
                #         print(f'Create mod skip year: Setting result to: Sales*Partial: {result}')
                #     sales_index_start +=1
                #     create_mod_partial_year = 0
                #     result = None
                #     x +=1
                    
                #     break
                else:
                    # print(f'Future_Sales_' + str(int(starting_year) + i))
                    # print(sales_val)
                    result = sales_val
                    df.loc[row.Index, f'Inventory_Target_' + str(int(starting_year) + i)] = int((int(sales_val) * 0.5) + (int(sales_val) * 0.16))
                    df.loc[row.Index, f'Future_Sale_Base_' + str(int(starting_year) + i)] = int(sales_val)
                    df.loc[row.Index, f'Future_Bulk_' + str(int(starting_year) + i)] = int(result)
                    # print(f'off-ideal-check - Create mod skip year: Setting result to: {result}')
                    result = None
                    sales_index_start += 1
                    x += 1
                    break
# Identify all future bulk years from column names
def calculate_runout_release_and_graph(df, current_year):
    future_years = sorted([
        int(col.split('_')[-1])
        for col in df.columns
        if col.startswith('Future_Bulk_')
    ])

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
        actual_inventory_col = f'Actual_Inventory_{year}'  # Adjust if needed

        if all(col in df.columns for col in [bulk_col, sale_col, actual_prev_col, ideal_prev_col]):
            # Convert numeric columns
            bulk = pd.to_numeric(df[bulk_col], errors='coerce').fillna(0)
            sale = pd.to_numeric(df[sale_col], errors='coerce').fillna(0)

            
            # --- Planned months (float) ---
            planned_months_float = np.where(sale > 0, bulk * 12.0 / sale, 0.0)

            # --- ROUND UP (ceiling) to integer months for both display & date math ---
            months_int = np.where(sale > 0, np.ceil(planned_months_float), 0).astype(np.int64)

            # Persist runout months and Off_Ideal
            df[runout_col] = months_int
            df[off_ideal_col] = 0

            # --- Actual runout date: add integer months to previous actual date ---
            actual_dates = pd.to_datetime(df[actual_prev_col], errors='coerce')
            df[actual_col] = [
                (d + pd.DateOffset(months=int(m))) if (pd.notnull(d) and pd.notnull(m)) else pd.NaT
                for d, m in zip(actual_dates, months_int)
            ]

            # --- Ideal release date: +12 months to previous ideal date ---
            ideal_dates = pd.to_datetime(df[ideal_prev_col], errors='coerce')
            df[ideal_col] = [
                (d + pd.DateOffset(months=12)) if pd.notnull(d) else pd.NaT
                for d in ideal_dates
            ]

            # --- Graph volume ---
            if year <= current_year and actual_inventory_col in df.columns:
                df[graph_col] = df[actual_inventory_col]
            else:
                df[graph_col] = bulk
        else:
            print(f"Missing columns for year {year}")

    return df
display.display(df)

df = calculate_runout_release_and_graph(df, 2025)

display.display(df)

# %%
#Future_Sale_Base2025

#Generates Barrel Information + NV

future_bulk = [col for col in df.columns if col.startswith("Future_Bulk_")]

df.fillna(805,inplace=True)
for future_col in future_bulk:
    year = create_year_string(future_col)
    get_index = df.columns.get_loc(future_col)
    barrel_multiplier = 23
    
    create_non_vintage_for_subtraction = df[f'{future_col}'].astype(float) * (df['nv_perc']/100).astype(float)
    create_bulk_for_subtraction = df[f'{future_col}'].astype(float) * (df['bulk_percentage']/100).astype(float)
    grape_fce_need = df[future_col].astype(float) - create_non_vintage_for_subtraction.astype(float)
    df.insert(get_index +1, f'NV_Use_Up_{year}', create_non_vintage_for_subtraction.astype(int))
    df.insert(get_index +2, f'Bulk_Need_{year}', create_bulk_for_subtraction.astype(int))
    df.insert(get_index +3, f'Grape_Need_{year}', grape_fce_need.astype(int))
    adjunct_calc_fce = df[f'Grape_Need_{year}'] * (df['adjunct_percentage']/100)
    barrel_calc_fce = df[f'Grape_Need_{year}'] * (df['barrel_percentage']/100)
    barrel_calc_bbl = (df[f'Grape_Need_{year}'] * (df['barrel_percentage']/100))/barrel_multiplier
    insert_calc_fce = df[f'Grape_Need_{year}'] * (df['insert_percentage']/100)
    insert_calc_bbl = (df[f'Grape_Need_{year}'] * (df['insert_percentage']/100))/barrel_multiplier
    stainless_calc_fce = df[f'Grape_Need_{year}'] * (df['stainless_percentage']/100)
    fo_calc_bbl = (df[f'Grape_Need_{year}'] * (df['fo_percentage']/100)/barrel_multiplier)
    ao_calc_bbl = (df[f'Grape_Need_{year}'] * (df['ao_percentage']/100)/barrel_multiplier)
    neutral_calc_bbl = (df[f'Grape_Need_{year}'] * (df['neutral_percentage']/100)/barrel_multiplier)
    block_adjunct_fce = df[f'Grape_Need_{year}'] * (df['stainlessBlock_percentage']/100)
    def cull_creator():
        pass
    
    df.insert(get_index +4, f'Total_Barrel_FCE_Need_{year}', barrel_calc_fce.astype(int))
    df.insert(get_index +5, f'Total_Barrel_BBL_Need_{year}', barrel_calc_bbl.astype(int))
    df.insert(get_index +6, f'FO_BBL_Need_{year}', fo_calc_bbl.astype(int))
    df.insert(get_index +7, f'AO_BBL_Need_{year}', ao_calc_bbl.astype(int))
    df.insert(get_index +8, f'Neutral_BBL_Need_{year}', neutral_calc_bbl.astype(int))
    df.insert(get_index +9, f'Insert_Barrel_FCE_Need_{year}', insert_calc_fce.astype(int))
    df.insert(get_index +10, f'Insert_Barrel_BBL_Need_{year}', insert_calc_bbl.astype(int))
    df.insert(get_index +11, f'Stainless_FCE_Need_{year}', stainless_calc_fce.astype(int))
    df.insert(get_index +12, f'Adjunct_FCE_Need_{year}', adjunct_calc_fce.astype(int))
    df.insert(get_index +13, f'Block_FCE_Need_{year}', block_adjunct_fce.astype(int))


    
    
# df.to_csv("Scenario_Anna_Numbers_10172025.csv")

display.display(df)


# %%
# --- Helper: sanitize filenames ---
def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in str(name)).strip().replace(" ", "_")


# --- Helper: dynamically ensure left margin fits y-tick descriptors ---
def ensure_left_margin(ax, pad_pts=10, min_left=0.06, max_left=0.35):
    """
    Dynamically adjust the figure's left margin so the y-axis tick labels
    are fully visible when saving/printing.
    """
    fig = ax.figure
    fig.canvas.draw()  # ensure text extents are computed
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


# --- Core chart function: plots and saves one chart for a given subset ---
def plot_runout_chart_for_subset(
    df_sub: pd.DataFrame,
    outfile: str,
    title: str,
    max_year: int = 2026,
    ideal_mode: str = "future",
    current_year: int = 2025,
    dpi: int = 300,
    filter_status_active: bool = False,  # <--- unfiltered by default per your request
):
    """
    Build and save a landscape runout chart for a given dataframe subset (Brand or Varietal).
    Returns True if a chart was saved, False if subset had no usable data.
    """
    if df_sub.empty:
        return False

    # Optional status filter inside the chart
    filtered_df = df_sub.copy()
    if filter_status_active and "Status" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Status"] == "Active"].copy()
        if filtered_df.empty:
            return False

    # Convert available date-like columns
    date_cols = [c for c in filtered_df.columns if c.startswith(("Actual_Runout_", "Ideal_Release_"))] + ["Current_date"]
    date_cols = [c for c in date_cols if c in filtered_df.columns]
    if date_cols:
        filtered_df[date_cols] = filtered_df[date_cols].apply(pd.to_datetime, errors="coerce")

    # Detect vintages and cap to max_year
    years_all = sorted({int(c.split("_")[-1]) for c in filtered_df.columns if c.startswith("Actual_Runout_")})
    years = [y for y in years_all if y <= max_year]
    if not years:
        return False

    # Axis extent (min current_date to Dec 31 max_year)
    min_date_all = filtered_df["Current_date"].min() if "Current_date" in filtered_df.columns else None
    max_date_series = filtered_df[[f"Actual_Runout_{y}" for y in years]].max(axis=1) if years else pd.Series(dtype="datetime64[ns]")
    max_date_data = max_date_series.max() if not max_date_series.empty else None
    max_date_cap = datetime(max_year, 12, 31)
    xmax_date = min(max_date_data, max_date_cap) if pd.notnull(max_date_data) else max_date_cap

    # --- Plot Settings ---
    ann_fontsize = 8
    ann_color = "white"
    ann_ha = "center"
    ann_va = "center"
    cmap = plt.get_cmap("tab10")
    color_map = {year: cmap(i % cmap.N) for i, year in enumerate(years)}

    # Landscape US Letter: width=11, height dynamic but capped at 8.5
    dynamic_height = min(len(filtered_df) * 0.6 + 1, 8.5)
    fig, ax = plt.subplots(figsize=(11, dynamic_height), constrained_layout=False)

    # --- Draw Bars ---
    for i, row in enumerate(filtered_df.itertuples()):
        drawn_years = set()
        current = getattr(row, "Current_date", None)

        # First segment start among capped years
        start_index = None
        for idx, y in enumerate(years):
            run = getattr(row, f"Actual_Runout_{y}", None)
            if pd.notnull(run) and (current is None or current < run):
                start_index = idx
                break

        if start_index is not None:
            # First segment (clip to cap)
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

                        # Annotation logic
                        if first <= current_year:
                            text = f"RO:{getattr(row, 'Runout_Months_' + str(first), '')}, OI:{getattr(row, 'Off_Ideal_' + str(first), '')}"
                        else:
                            text = f"Vol:{int(graph_val/1000)}K FCE"
                        ax.text(start_num + width / 2, i, text, color=ann_color, fontsize=ann_fontsize, ha=ann_ha, va=ann_va)

            # Subsequent segments (clip to cap)
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
                    text = f"RO:{getattr(row, 'Runout_Months_' + str(ny), '')}, OI:{getattr(row, 'Off_Ideal_' + str(ny), '')}"
                else:
                    text = f"Vol:{int(graph_val_next/1000)}K FCE"
                ax.text(s_num + width_next / 2, i, text, color=ann_color, fontsize=ann_fontsize, ha=ann_ha, va=ann_va)

        # Ideal markers (black dotted, clipped to cap)
        for y in drawn_years:
            ideal = getattr(row, f"Ideal_Release_{y}", None)
            if pd.notnull(ideal):
                if ideal_mode == "future" and current is not None and ideal <= current:
                    continue
                ideal_num = mdates.date2num(min(ideal, max_date_cap))
                ax.vlines(ideal_num, i - 0.2, i + 0.2, linestyles="dotted", colors="black")
                label = f"(V{y} - {mdates.num2date(ideal_num).strftime('%b %Y')})"
                ax.text(ideal_num, i + 0.25, label, color="black", ha="center", va="bottom", fontsize=8)

    # --- Axis & Legend ---
    min_num = mdates.date2num(min_date_all) if pd.notnull(min_date_all) else mdates.date2num(datetime(current_year, 1, 1))
    max_num = mdates.date2num(xmax_date)
    ax.set_xlim(min_num, max_num)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlabel("Calendar Date", fontsize=12, labelpad=6)
    plt.xticks(rotation=30)

    
    ax.set_yticks(range(len(filtered_df)))

    # Build labels that include Brand when available
    if "Brand" in filtered_df.columns:
        # Normalize possible NaNs/empties
        desc = filtered_df["description"].fillna("").astype(str)
        brand = filtered_df["Brand"].fillna("").astype(str)

        # Combine: "Description — Brand" when Brand is non-empty; else just "Description"
        y_labels = [
            f"{d}|{b}" if b.strip() else d
            for d, b in zip(desc, brand)
        ]
    else:
        # Fallback to description or index
        y_labels = (
            filtered_df["description"].astype(str)
            if "description" in filtered_df.columns
            else filtered_df.index.astype(str).tolist()
        )

    ax.set_yticklabels(y_labels)


    legend_handles = [mpatches.Patch(color=color_map[y], label=str(y)) for y in years]
    ax.legend(handles=legend_handles, title="Vintage", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, frameon=False)

    ax.margins(x=0.01, y=0.02)
    ensure_left_margin(ax, pad_pts=10, min_left=0.06, max_left=0.35)

    plt.title(title)
    
    outfile_path = Path(outfile)
    outfile_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(outfile_path), dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return True



# --- Driver: create folders and save charts per Varietal and Brand ---
def save_charts_by_brand_and_varietal(
    df: pd.DataFrame,
    base_path: Path,
    folder_name: str,
    varietal_subfolder: str = "Varietal",
    brand_subfolder: str = "Brand",
    max_year: int = 2028,
    ideal_mode: str = "future",
    current_year: int = 2025,
    dpi: int = 300,
    filter_status_active: bool = False,  # keep unfiltered unless you set True
):
    """
    Generates PNG charts for each Varietal and each Brand under:
    base_path / folder_name / Varietal
    base_path / folder_name / Brand
    """
    
    
    
    scenario_path = base_path / folder_name
    varietal_dir = scenario_path / varietal_subfolder
    brand_dir = scenario_path / brand_subfolder
    varietal_dir.mkdir(parents=True, exist_ok=True)
    brand_dir.mkdir(parents=True, exist_ok=True)

    saved_varietal, saved_brand = 0, 0

    # --- Per Varietal ---
    if "Varietal" in df.columns:
        varietals = sorted([v for v in df["Varietal"].dropna().unique()])
        for var in varietals:
            subset = df[df["Varietal"] == var].copy()
            outfile = varietal_dir / f"{sanitize_filename(var)}.png"
            title = f"Runout – Varietal: {var} (≤ {max_year})"
            ok = plot_runout_chart_for_subset(
                subset,
                outfile=str(outfile),
                title=title,
                max_year=max_year,
                ideal_mode=ideal_mode,
                current_year=current_year,
                dpi=dpi,
                filter_status_active=filter_status_active,
            )
            if ok:
                saved_varietal += 1

    # --- Per Brand ---
    if "Brand" in df.columns:
        brands = sorted([b for b in df["Brand"].dropna().unique()])
        for brand in brands:
            subset = df[df["Brand"] == brand].copy()
            outfile = brand_dir / f"{sanitize_filename(brand)}.png"
            title = f"Runout – Brand: {brand} (≤ {max_year})"
            ok = plot_runout_chart_for_subset(
                subset,
                outfile=str(outfile),
                title=title,
                max_year=max_year,
                ideal_mode=ideal_mode,
                current_year=current_year,
                dpi=dpi,
                filter_status_active=filter_status_active,
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

# --- Example usage ---
# df = calculate_runout_release_and_graph(df, current_year=2025)  # if not already computed upstream
summary = save_charts_by_brand_and_varietal(
    df=df,
    base_path=base_path,
    folder_name=folder_name,
    varietal_subfolder=varietal_subfolder,
    brand_subfolder=brand_subfolder,
    max_year=2028,           # cap the timeline for clarity
    ideal_mode="future",     # show only future ideal markers
    current_year=2025,       # controls annotation detail logic
    dpi=300,
    filter_status_active=True  # <-- unfiltered as requested
)
print(summary)


# %%
#Top_50_Export_Section - created top 50 sorted by first sales column.
off_ideal = [col for col in df.columns if col.startswith("Off_Ideal_")]
df_top_50 = df[['Brand','Concat','description','Varietal', 'Roll_Up', 'Color', off_ideal[-1], sales_plan_cols[0]]]
df_top_50_sorted = df_top_50.sort_values(sales_plan_cols[0], ascending=False)
display.display(df_top_50_sorted.nlargest(50, sales_plan_cols[0]))

# %%
#Create Inventory By Year current with pre-sort by color
df_inventory_group = df.groupby(['Color','Roll_Up'])[inv_plan_cols].sum()
display.display(df_inventory_group)

# %%
#Creates Dataframe for use later
df_report = df[['Brand','Concat','description','Varietal', 'Roll_Up', 'Color', 'Future_Bulk_2026', 'NV_Use_Up_2026', 'Grape_Need_2026', 'Future_Bulk_2027', 'NV_Use_Up_2027', 'Grape_Need_2027', 'Future_Bulk_2028', 'NV_Use_Up_2028', 'Grape_Need_2028', 'Future_Bulk_2029', 'NV_Use_Up_2029', 'Grape_Need_2029', 'Future_Bulk_2030', 'NV_Use_Up_2030', 'Grape_Need_2030']]


# %%
#Creates just sales columns by item and roll up
first_index_sales = df.columns.get_loc(sales_plan_cols[0])
last_index_sales = df.columns.get_loc(sales_plan_cols[-1])

df_fiscal_1 = df.loc[:,['Concat','description' ,'Roll_Up', 'Color']]
df_fiscal_2_cases = df.iloc[:,first_index_sales:last_index_sales+1]


df_fiscal_pre_total_cases = [df_fiscal_1, df_fiscal_2_cases]
df_fiscal_total_cases = pd.concat(df_fiscal_pre_total_cases, axis=1).reindex(df_fiscal_1.index)
df_report_sales_cases = df_fiscal_total_cases.groupby(['Roll_Up'])[sales_plan_cols].sum()
df_report_sales_color = df_fiscal_total_cases.groupby(['Color'])[sales_plan_cols].sum()
display.display(df_report_sales_color)

def adjust_sales(row):
    rollup_value = row['Roll_Up']
    if rollup_value == 'RSL':
        divisor = 60
    else:
        divisor = 65
    
    for col in sales_plan_cols:
        row[col] = row[col]/divisor
    
    return row

df_fiscal_total_tons = df_fiscal_total_cases.apply(adjust_sales, axis = 1)

df_report_sales_tons = df_fiscal_total_tons.groupby(['Roll_Up'])[sales_plan_cols].sum()
df_report_sales_color_cases = df_fiscal_total_tons.groupby(['Color', 'Roll_Up'])[sales_plan_cols].sum()
display.display(df_report_sales_color_cases)

# %% [markdown]
# 

# %%
grape_fce_need = [col for col in df_report.columns if col.startswith("Future_Bulk_") or col.startswith('NV_Use_Up') or col.startswith("Grape_Need_")]
def adjust_grape(row):
    rollup_value = row['Roll_Up']
    if rollup_value == 'RSL':
        divisor = 60
    else:
        divisor = 65
    
    for col in grape_fce_need:
        row[col] = row[col]/divisor
    
    return row
df_report.groupby(['Color','Roll_Up'])[grape_fce_need].sum()
df_report_tons_base = df_report.apply(adjust_grape, axis = 1)
df_report_tons = df_report_tons_base.groupby(['Color', 'Roll_Up'])[grape_fce_need].sum()
display.display(df_report_tons)

# %%
# csv_file = r'C:\Users\chh03\OneDrive - SMWE\Code\Scripts\VintageReq\Base_Data_Allocation_5.9.csv'
# dfwio = pd.read_csv(csv_file, encoding='latin-1')

# dfwio.head

# dfwio['Tons_Anticipated'] = dfwio['Contract Acres'] * dfwio['Committed TPA 2026']

# dfwio.groupby(['Contract Expires After'])[['Tons_Anticipated']].sum()


