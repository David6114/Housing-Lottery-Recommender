import requests

import numpy as np
import pandas as pd


APARTMENT_SIZES = (80, 100, 125)
VAT_RATE = 0.18


def extract_project_list(json_data):
    """
    Recursively and silently searches the JSON response for the project list.

    Parameters
    ----------
    json_data : dict or list
        The raw parsed JSON from the API.

    Returns
    -------
    list
        The extracted list of projects, or an empty list if not found.
    """
    if isinstance(json_data, list):
        return json_data

    if isinstance(json_data, dict):
        # 1. Look for a list of dictionaries at the current level
        for value in json_data.values():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                return value

        # 2. Dig one level deeper if not found
        for value in json_data.values():
            if isinstance(value, dict):
                nested_result = extract_project_list(value)
                if nested_result:
                    return nested_result

    return []


def fetch_lottery_data(url):
    """
    Fetches lottery data from the given URL.

    Parameters
    ----------
    url : str
        The endpoint URL pointing to the Mechir Lamishtaken JSON API.

    Returns
    -------
    list
        The parsed and extracted list of projects.
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        raw_json = response.json()

        return extract_project_list(raw_json)

    except (requests.exceptions.RequestException, ValueError):
        return []


def clean_dataframe(df):
    """
    Cleans the DataFrame by removing completely empty columns and
    columns with zero variance (identical values across all rows).

    Parameters
    ----------
    df : pandas.DataFrame
        The raw DataFrame containing lottery data.

    Returns
    -------
    pandas.DataFrame
        The cleaned DataFrame.
    """
    if df.empty:
        return df

    # Drop columns that are completely empty
    df_cleaned = df.dropna(axis=1, how='all')

    # Drop columns with zero variance
    cols_to_keep = []
    for col in df_cleaned.columns:
        try:
            if df_cleaned[col].nunique(dropna=False) > 1:
                cols_to_keep.append(col)
        except TypeError:
            if df_cleaned[col].astype(str).nunique() > 1:
                cols_to_keep.append(col)

    df_cleaned = df_cleaned[cols_to_keep]
    return df_cleaned


def extract_discount_features(df):
    """
    Extracts numerical discount information from the Notes column using regex,
    with a fallback to PricePerUnit if the updated price is missing.

    Parameters
    ----------
    df : pandas.DataFrame
        The cleaned DataFrame containing 'Notes' and 'PricePerUnit' columns.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with 3 new extracted numerical columns for Phase 2.
    """
    if 'Notes' not in df.columns:
        return df

    # Fill NaN values with an empty string to avoid regex errors
    notes_clean = df['Notes'].fillna('')

    # 1. Clean HTML tags (e.g., <p dir="RTL">, <span>)
    notes_clean = notes_clean.str.replace(r'<[^>]*>', ' ', regex=True)

    # 2. Clean HTML entities
    notes_clean = notes_clean.str.replace('&nbsp;', ' ', regex=False)
    notes_clean = notes_clean.str.replace('&quot;', '"', regex=False)

    # 3. Extract Discount Percentage (e.g., '25%')
    percent_series = notes_clean.str.extract(r'(\d{1,2})%')[0]
    df['Discount_Percentage'] = pd.to_numeric(percent_series, errors='coerce').apply(
        lambda x: x / 100 if x > 1 else x
    )

    # 4. Extract Max Discount Cap (e.g., after 'ב.' or 'עד') - removes commas
    cap_series = notes_clean.str.extract(r'(?:ב\.|עד)\s*([0-9]{1,3}(?:,?[0-9]{3})*)')[0]
    cap_series = cap_series.str.replace(',', '', regex=False)
    df['Max_Discount_Cap'] = pd.to_numeric(cap_series, errors='coerce')

    # 5. Extract Updated Price Per Sqm (e.g., 'בסך 20,753 ₪ למ"ר') - removes commas
    price_sqm_series = notes_clean.str.extract(r'בסך\s*([0-9]{1,3}(?:,?[0-9]{3})*)\s*₪?\s*למ"ר')[0]
    price_sqm_series = price_sqm_series.str.replace(',', '', regex=False)
    df['Updated_Price_Per_Sqm'] = pd.to_numeric(price_sqm_series, errors='coerce')

    # 6. Fallback Logic: If Updated_Price_Per_Sqm is missing, use PricePerUnit
    if 'PricePerUnit' in df.columns:
        df['Updated_Price_Per_Sqm'] = df['Updated_Price_Per_Sqm'].fillna(df['PricePerUnit'])

    return df


def calculate_apartment_costs(df, apartment_sizes=APARTMENT_SIZES, vat_rate=VAT_RATE):
    """
    Calculates the exact pricing, discount parameters, and total savings
    for specific apartment sizes, based on the Mechir Matara discount rules.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing project-level data with base and updated prices.
    apartment_sizes : list
        A list of apartment sizes (in sqm) to calculate the metrics for.
    vat_rate : float
        The VAT rate to apply to the prices per square meter.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with 7 new calculated columns for each apartment size.
    """
    try:
        df_calc = df.copy()

        discount_pct = df_calc['Discount_Percentage']

        # Fill missing caps with infinity so they don't break the minimum calculation
        max_discount_cap = df_calc['Max_Discount_Cap'].fillna(float('inf'))

        for size in apartment_sizes:
            # 1. Original apartment price including VAT (based on PricePerUnit)
            col_orig_price = f'Original_Price_{size}sqm'
            df_calc[col_orig_price] = df_calc['PricePerUnit'] * size * (1 + vat_rate)

            # 2. Updated apartment price including VAT (based on Updated_Price_Per_Sqm)
            col_updated_price = f'Updated_Price_{size}sqm'
            df_calc[col_updated_price] = df_calc['Updated_Price_Per_Sqm'] * size * (1 + vat_rate)

            # 3. Apartment price after percentage discount (without discount cap)
            col_price_after_pct_disc = f'Price_After_Pct_Discount_{size}sqm'
            df_calc[col_price_after_pct_disc] = df_calc[col_updated_price] * (1 - discount_pct)

            # 4. Theoretical discount amount (based on percentage only)
            col_calculated_disc = f'Calculated_Discount_{size}sqm'
            df_calc[col_calculated_disc] = df_calc[col_updated_price] * discount_pct

            # 5. Actual discount amount (capped at the max discount cap, e.g., 600,000)
            col_final_disc = f'Final_Discount_{size}sqm'
            df_calc[col_final_disc] = np.minimum(df_calc[col_calculated_disc], max_discount_cap)

            # 6. Final updated apartment price
            # Safety net: The minimum between the original price (no discount)
            # and the updated price minus the actual discount
            col_final_price = f'Final_Price_{size}sqm'
            price_after_capped_disc = df_calc[col_updated_price] - df_calc[col_final_disc]
            df_calc[col_final_price] = np.minimum(df_calc[col_orig_price], price_after_capped_disc)

            # 7. The bottom line - Final discount/savings amount
            # (the difference between full price and the price to pay)
            col_total_savings = f'Total_Savings_{size}sqm'
            df_calc[col_total_savings] = df_calc[col_updated_price] - df_calc[col_final_price]

            col_total_savings_pct = f'Total_Savings_{size}sqm_pct'
            df_calc[col_total_savings_pct] = df_calc[col_total_savings] / df_calc[col_final_price]

        return df_calc

    except Exception as e:
        print(f'Error calculating apartment costs: {e}')
        return df


def calculate_city_probabilities(df_fin, apartment_sizes=APARTMENT_SIZES):
    """
    Calculates city-level probabilities, average costs, and expected savings.

    Parameters
    ----------
    df_fin : pandas.DataFrame
        The DataFrame from Phase 1, AFTER passing through calculate_apartment_costs.
    apartment_sizes : list
        A list of apartment sizes (in sqm) to calculate expected values for.

    Returns
    -------
    pandas.DataFrame
        The final aggregated Phase 2 DataFrame with probabilities and expected values.
    """
    try:
        # Step 1: Define the base aggregations for supply and demand
        agg_kwargs = {
            'Total_City_Supply': ('LotteryApparmentsNum', 'sum'),
            'LocalHousing': ('LocalHousing', 'sum'),
            'Handicapped_Supply': ('HousingUnitsForHandicapped', 'sum'),
            'Reservists_Supply': ('HU_Reservists_L', 'sum'),
            'Combat_Reservists_Supply': ('HU_CombatReservist_L', 'sum'),
            'Max_Subscribers': ('TotalSubscribers', 'max'),
            'Max_Local_Subscribers': ('TotalLocalSubscribers', 'max')
        }

        # Step 2: Dynamically add aggregations for prices and savings (calculating the MEAN per city)
        for size in apartment_sizes:
            agg_kwargs[f'Avg_Final_Price_{size}sqm'] = (f'Final_Price_{size}sqm', 'mean')
            agg_kwargs[f'Avg_Total_Savings_{size}sqm'] = (f'Total_Savings_{size}sqm', 'mean')
            agg_kwargs[f'Avg_Total_Savings_{size}sqm_pct'] = (f'Total_Savings_{size}sqm_pct', 'mean')

        # Step 3: Group by city and apply all aggregations
        city_stats = df_fin.groupby('CityDescription').agg(**agg_kwargs).reset_index()

        # Handle missing values
        city_stats = city_stats.fillna(0)

        # Step 4: Calculate Reserved Supply
        city_stats['Reserved_Supply'] = (
                city_stats['LocalHousing'] +
                city_stats['Handicapped_Supply'] +
                city_stats['Reservists_Supply'] +
                city_stats['Combat_Reservists_Supply']
        )

        # Step 5: Calculate General Probability (P_General)
        city_stats['General_Supply'] = (city_stats['Total_City_Supply'] - city_stats['Reserved_Supply']).clip(lower=0)
        city_stats['Effective_General_Demand'] = (city_stats['Max_Subscribers'] - city_stats['Reserved_Supply']).clip(
            lower=1)
        city_stats['Probability_General'] = (
                    city_stats['General_Supply'] / city_stats['Effective_General_Demand']).clip(upper=1.0)

        # Step 6: Calculate Local Probability (P_Total_Local)
        city_stats['Effective_Local_Demand'] = city_stats['Max_Local_Subscribers'].clip(lower=1)
        city_stats['Probability_Local_Draw'] = (city_stats['LocalHousing'] / city_stats['Effective_Local_Demand']).clip(
            upper=1.0)

        # Cascading Probability: P(Local) + P(Not Local) * P(General)
        city_stats['Probability_Total_Local'] = city_stats['Probability_Local_Draw'] + (
                (1 - city_stats['Probability_Local_Draw']) * city_stats['Probability_General']
        ).clip(upper=1.0)

        # Step 7: The Holy Grail - Calculate Expected Savings
        for size in apartment_sizes:
            avg_savings_col = f'Avg_Total_Savings_{size}sqm'

            # Expected profit for a general applicant
            # (general probability multiplied by the average discount in the city)
            col_expected_general = f'Expected_Savings_General_{size}sqm'
            city_stats[col_expected_general] = city_stats['Probability_General'] * city_stats[avg_savings_col]

            col_expected_general_pct = f'Expected_Savings_General_{size}sqm_pct'
            city_stats[col_expected_general_pct] = city_stats[col_expected_general] / city_stats[
                f'Avg_Final_Price_{size}sqm']

            # Expected profit for a local applicant
            # (combined probability multiplied by the average discount in the city)
            col_expected_local = f'Expected_Savings_Local_{size}sqm'
            city_stats[col_expected_local] = city_stats['Probability_Total_Local'] * city_stats[avg_savings_col]

            col_expected_local_pct = f'Expected_Savings_Local_{size}sqm_pct'
            city_stats[col_expected_local_pct] = city_stats[col_expected_local] / city_stats[
                f'Avg_Final_Price_{size}sqm']

        # Sort the final DataFrame by the expected savings for a general 100sqm apartment
        # (This brings the absolute best financial opportunities to the top!)
        city_stats = city_stats.sort_values(by='Expected_Savings_General_100sqm', ascending=False)

        return city_stats

    except Exception as e:
        print(f'Error calculating city statistics: {e}')
        return pd.DataFrame()


def run_etl_pipeline(url):
    """
    Executes the full Phase 1 ETL pipeline: Fetch, Convert, and Clean.

    Parameters
    ----------
    url : str
        The URL of the API.

    Returns
    -------
    pandas.DataFrame
        The fully processed and cleaned DataFrame ready for Phase 2.
    """
    raw_data = fetch_lottery_data(url)

    if not raw_data:
        return pd.DataFrame()

    try:
        df_raw = pd.DataFrame(raw_data)
    except Exception:
        return pd.DataFrame()

    df_clean = clean_dataframe(df_raw)
    df_with_features = extract_discount_features(df_clean)
    df_with_prices = calculate_apartment_costs(df_with_features)

    return df_with_prices


if __name__ == '__main__':
    api_url = (
        'https://dira.moch.gov.il/api/Invoker?method=Projects&param='
        '%3FfirstApplicantIdentityNumber%3D%26secondApplicantIdentityNumber%3D'
        '%26ProjectStatus%3D4%26Entitlement%3D1%26PageNumber%3D1%26PageSize%3D100%26IsInit%3Dfalse%26'
    )

    print('Running ETL Pipeline...')
    df_final = run_etl_pipeline(api_url)
    city_stats = calculate_city_probabilities(df_final)

    if not df_final.empty:
        df_final.to_excel('output.xlsx', index=False)
        city_stats.to_excel('cities.xlsx', index=False)
        print(f'Success! Cleaned DataFrame shape: {df_final.shape}')
        print(df_final.head())

    else:
        print('The DataFrame is empty. Please check the URL or network.')
