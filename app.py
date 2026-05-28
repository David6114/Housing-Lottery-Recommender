import re
import numpy as np
import pandas as pd
import streamlit as st
# Import phase 1 & 2 pipeline functions from your etl module
from etl import run_etl_pipeline, calculate_city_probabilities


@st.cache_data(ttl=600)
def load_city_data():
    """
    Fetches live project data and aggregates it dynamically to city-level statistics.

    Returns
    -------
    pandas.DataFrame
        The aggregated city statistics dataframe, or empty dataframe on failure.
    """
    api_url = 'https://dira.moch.gov.il/api/Invoker?method=Projects&param=%3FfirstApplicantIdentityNumber%3D%26secondApplicantIdentityNumber%3D%26ProjectStatus%3D4%26Entitlement%3D1%26PageNumber%3D1%26PageSize%3D100%26IsInit%3Dfalse%26'

    try:
        # Run the live project-level ETL pipeline from Phase 1 & 2
        df_projects = run_etl_pipeline(api_url)

        if df_projects.empty:
            return pd.DataFrame()

        # Dynamically compute city-level statistics and probabilities
        df_cities = calculate_city_probabilities(df_projects)
        return df_cities

    except Exception:
        return pd.DataFrame()


# Configure the web page for a mobile-first, centered layout
st.set_page_config(
    page_title='Housing Lottery Recommender',
    layout='centered',
    initial_sidebar_state='collapsed'
)

# Application Header
st.title('Housing Lottery Recommender')
st.markdown('This system ranks Israeli housing lottery cities based on live odds and financial metrics.')

# Load the city statistics dynamically via the cached pipeline
df_cities = load_city_data()

if df_cities.empty:
    st.error('Failed to load lottery data. Please check the API or network connection.')
else:
    # --- Step 1: Extract Available Apartment Sizes Dynamically ---
    available_sizes = []
    for col in df_cities.columns:
        match = re.search(r'Expected_Savings_General_(\d+)sqm$', col)
        if match:
            available_sizes.append(int(match.group(1)))

    # Remove duplicates and sort the extracted sizes
    available_sizes = sorted(list(set(available_sizes)))

    if not available_sizes:
        st.error('No apartment sizes found in the data pipeline.')
    else:
        # --- Compact Input Section (Single Row Layout) ---
        st.subheader('👤 User Profile & Configuration')

        # Extract unique cities for the local residency selection widget
        unique_cities = sorted(df_cities['CityDescription'].unique())

        # Creating a 3-column horizontal row to save vertical space
        input_col1, input_col2, input_col3 = st.columns([2, 1, 1.2])

        with input_col1:
            local_cities = st.multiselect(
                'Local Resident Cities:',
                options=unique_cities,
                max_selections=2,
                placeholder='Select up to 2'
            )

        with input_col2:
            selected_size = st.selectbox(
                'Size (Sqm):',
                options=available_sizes,
                index=0
            )

        with input_col3:
            # Sorting preference used for immediate Top 3 calculations
            sort_options = ['Winning Probability', 'Expected Savings (ILS)', 'Expected Savings (%)']
            chosen_sort = st.selectbox(
                'Sort Metrics By:',
                options=sort_options,
                index=0
            )

        # --- Dynamic Simulation & Processing Logic ---
        df_display = df_cities.copy()

        # Determine which rows match the user's local resident cities
        is_local_city = df_display['CityDescription'].isin(local_cities)

        # 1. Probability and Expected Values (Dynamic based on Simulation Status)
        df_display['Winning Probability'] = np.where(
            is_local_city,
            df_display['Probability_Total_Local'],
            df_display['Probability_General']
        )

        df_display['Expected Savings (ILS)'] = np.where(
            is_local_city,
            df_display[f'Expected_Savings_Local_{selected_size}sqm'],
            df_display[f'Expected_Savings_General_{selected_size}sqm']
        )

        df_display['Expected Savings (%)'] = np.where(
            is_local_city,
            df_display[f'Expected_Savings_Local_{selected_size}sqm_pct'],
            df_display[f'Expected_Savings_General_{selected_size}sqm_pct']
        )

        df_display['Application Status'] = np.where(
            is_local_city,
            'Local Resident',
            'General Public'
        )

        # 2. Absolute Real Values (Fixed attributes for the chosen apartment size in that city)
        df_display['Final Price (ILS)'] = df_display[f'Avg_Final_Price_{selected_size}sqm']
        df_display['Final Discount (ILS)'] = df_display[f'Avg_Total_Savings_{selected_size}sqm']
        df_display['Final Discount (%)'] = df_display[f'Avg_Total_Savings_{selected_size}sqm_pct']

        # Sort dataframe descending based on the user's preferred metric
        df_ranked = df_display.sort_values(by=chosen_sort, ascending=False)

        # Map internally computed columns to user-friendly presentation headers
        display_columns = {
            'CityDescription': 'City',
            'Application Status': 'Simulated Status',
            'Winning Probability': 'Winning Probability',
            'Final Price (ILS)': 'Final Price (ILS)',
            'Final Discount (ILS)': 'Final Discount (ILS)',
            'Final Discount (%)': 'Final Discount (%)',
            'Expected Savings (ILS)': 'Expected Savings (ILS)',
            'Expected Savings (%)': 'Expected Savings (%)',
        }

        df_final_table = df_ranked[list(display_columns.keys())].rename(columns=display_columns)

        # --- Top 3 Recommendations Section (Prominently at the Top) ---
        st.markdown('---')
        st.subheader('🏆 Top 3 Recommendations')

        if not df_final_table.empty:
            # Slice the top 3 items for summary presentation
            top_3_df = df_final_table.head(3)

            # Construct a clean, markdown list for bullet-proof mobile responsive layout
            summary_markdown = ''
            medals = {1: '🥇', 2: '🥈', 3: '🥉'}

            for idx, (index, row) in enumerate(top_3_df.iterrows(), 1):

                if chosen_sort == 'Winning Probability':
                    formatted_metric = f"{row['Winning Probability']:.2%}"
                elif chosen_sort == 'Expected Savings (ILS)':
                    formatted_metric = f"{row['Expected Savings (ILS)']:,.0f} ₪"
                else:
                    formatted_metric = f"{row['Expected Savings (%)']:.1%}"

                final_price_val = row['Final Price (ILS)']
                city_name = row['City']
                status_val = row['Simulated Status']

                # שליפת המדליה המתאימה, או ברירת מחדל אם יש יותר מ-3
                medal = medals.get(idx, '🏅')

                summary_markdown += (
                    f'{medal} **#{idx} {city_name}** ({status_val})  \n'
                    f'➔ *{chosen_sort}:* **{formatted_metric}** | *Final Price:* {final_price_val:,.0f} ₪ \n\n'
                )

            st.success(summary_markdown)

        # --- Complete Data Grid (At the Bottom) ---
        st.subheader('📊 Detailed Analysis Matrix')

        # Render the fully styled data frame below the top recommendations
        st.dataframe(
            df_final_table.style.format({
                'Winning Probability': '{:.2%}',
                'Expected Savings (ILS)': '{:,.0f} ₪',
                'Expected Savings (%)': '{:.1%}',
                'Final Price (ILS)': '{:,.0f} ₪',
                'Final Discount (ILS)': '{:,.0f} ₪',
                'Final Discount (%)': '{:.1%}'
            }),
            use_container_width=True,
            hide_index=True
        )