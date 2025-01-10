# Preprocessing RAW data & Postprocessing Predicted data
import pandas as pd
import numpy as np
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')

def convert_time_to_seconds(time_str):
    """Convert a time string like '20 h 56 m 9 s' to total seconds."""
    if isinstance(time_str, str):
        pattern = r"((?P<hours>\d+)\s*h)?\s*((?P<minutes>\d+)\s*m)?\s*((?P<seconds>\d+)\s*s)?"
        match = re.match(pattern, time_str)
        if not match:
            return np.nan
        parts = match.groupdict()
        hours = int(parts['hours']) if parts['hours'] else 0
        minutes = int(parts['minutes']) if parts['minutes'] else 0
        seconds = int(parts['seconds']) if parts['seconds'] else 0
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
    elif pd.isna(time_str):
        return 0
    else:
        return time_str

def graphing_prtg(file_path):
    df = pd.read_csv(file_path)

    # Ensure required columns are present
    required_columns = ['datetime', 'datetime_raw', 'uptimevalue_raw']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' is not present in the data")

    # Convert datetime to proper format
    df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %H:%M')

    # Convert uptimevalue_raw to numeric (seconds)
    df['uptimevalue_raw'] = df['uptimevalue_raw'].apply(convert_time_to_seconds)

    # Calculate Link Value
    df['Link Value'] = np.where(df['uptimevalue_raw'] > 0, 'OK', 'ERROR')
    df['Previous Link Value'] = df['Link Value'].shift(1)
    df['After Link Value'] = df['Link Value'].shift(-1)

    # Determine Status Link
    df['Status Link'] = np.where(
        (df['Previous Link Value'] == 'ERROR') & (df['Link Value'] == 'OK'), 'UP',
        np.where((df['After Link Value'] == 'ERROR') & (df['Link Value'] == 'OK'), 'DOWN', 'IGNORE')
    )

    # Calculate IndexTrue
    df['IndexTrue'] = df['Status Link'].apply(lambda x: 1 if x in ['UP', 'DOWN'] else 0).cumsum()

    # Calculate Selisih
    def calculate_selisih(row):
        if row['Status Link'] == 'UP' and row['IndexTrue'] == 1:
            return 0
        elif row['Status Link'] == 'DOWN':
            next_up = df[(df['IndexTrue'] > row['IndexTrue']) & (df['Status Link'] == 'UP') & (df['IndexTrue'] == row['IndexTrue'] + 1)]['uptimevalue_raw']
            if not next_up.empty:
                return next_up.iloc[0] - row['uptimevalue_raw']
        return 0

    df['Selisih'] = df.apply(calculate_selisih, axis=1)

    # Analyze the data (Analisis)
    df['Analisis'] = np.where(df['Selisih'] > 0, 'LINK', np.where(df['Selisih'] < 0, 'NON-LINK', ''))

    # Calculate Analisis Link Tahap 2
    def calculate_analisis_link_tahap_2(row):
        if row['Analisis'] == 'LINK':
            next_link = df[(df['IndexTrue'] > row['IndexTrue']) & (df['IndexTrue'] == row['IndexTrue'] + 1)]['datetime_raw']
            if not next_link.empty:
                return next_link.iloc[0] - row['datetime_raw']
        return np.nan

    df['Analisis Link Tahap 2'] = df.apply(calculate_analisis_link_tahap_2, axis=1)

    # Calculate Durasi Hidden Non-Link
    def calculate_durasi_hidden_non_link(row):
        if row['Analisis'] == 'LINK':
            next_uptime = df[(df['IndexTrue'] > row['IndexTrue']) & (df['IndexTrue'] == row['IndexTrue'] + 1)]['uptimevalue_raw']
            if not next_uptime.empty:
                return row['Analisis Link Tahap 2'] - (next_uptime.iloc[0] / 60)
        return 0

    df['Durasi Hidden Non-Link'] = df.apply(calculate_durasi_hidden_non_link, axis=1)

    # Calculate Durasi Real Restitusi
    def calculate_durasi_real_restitusi(row):
        if row['Analisis'] == 'LINK' and row['Durasi Hidden Non-Link'] > 0:
            next_uptime = df[(df['IndexTrue'] > row['IndexTrue']) & (df['IndexTrue'] == row['IndexTrue'] + 1)]['uptimevalue_raw']
            if not next_uptime.empty:
                return next_uptime.iloc[0]
        elif row['Analisis'] == 'LINK' and row['Durasi Hidden Non-Link'] < 0:
            return row['Selisih']
        return 0

    df['Durasi Real Restitusi'] = df.apply(calculate_durasi_real_restitusi, axis=1)

    # Dynamically calculate total period
    total_period = (df['datetime'].max() - df['datetime'].min()).total_seconds()

    # Calculate Service Level iteratively
    df['Service Level'] = 0.0
    df['Service Level'].iloc[0] = 1 - (df['Durasi Real Restitusi'].iloc[0] / total_period)

    for i in range(1, len(df)):
        df.loc[i, 'Service Level'] = df.loc[i - 1, 'Service Level'] - (df.loc[i, 'Durasi Real Restitusi'] / total_period)

    # Link/Non-Link Event PNG output

    # Ensure datetime column is in proper datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Initialize dictionaries to store data for LINK and NON-LINK events
    event_data_dict = {'LINK': [], 'NON-LINK': []}

    # Loop through each row to identify DOWN events and categorize based on Analysis column
    for index, row in df.iterrows():
        if row['Status Link'] == 'DOWN':
            # Capture 25 rows before the DOWN event
            before_down = df.iloc[max(0, index - 25):index]

            # Look for the corresponding UP event (end of the LINK event)
            up_event = df[(df.index > index) & (df['Status Link'] == 'UP')].head(1)
            if not up_event.empty:
                up_index = up_event.index[0]
                # Capture 25 rows after the UP event
                after_up = df.iloc[up_index + 1:min(up_index + 26, len(df))]

                # Combine before, during (DOWN to UP), and after rows
                during_event = df.iloc[index:up_index + 1]
                event_data = pd.concat([before_down, during_event, after_up])

                # Categorize the event based on the Analysis column of the DOWN row
                analysis_type = row['Analisis']  # Assumes Analysis column exists
                if analysis_type in event_data_dict:
                    event_data_dict[analysis_type].append(event_data)

    # Define the output directory for saving graphs
    output_dir = "graphs_prtg"
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Plot each category separately and save the graphs
    for analysis_type, events in event_data_dict.items():
        for i, event_data in enumerate(events):
            plt.figure(figsize=(12, 6))
            plt.plot(event_data['datetime'], event_data['uptimevalue_raw'], marker='o', label='Uptime During Event')

            # Add DOWN and UP event markers
            if len(event_data) > 25:
                plt.axvline(event_data.iloc[25]['datetime'], color='red', linestyle='--', label='DOWN Event')
            if len(event_data) > 51:  # Check to avoid indexing errors for UP event
                plt.axvline(event_data.iloc[len(event_data) - 26]['datetime'], color='green', linestyle='--', label='UP Event')

            plt.xlabel('Datetime')
            plt.ylabel('Uptime (seconds)')
            plt.title(f'{analysis_type} Event {i + 1}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Save the figure to a file
            output_path = os.path.join(output_dir, f"{analysis_type}_Event_{i + 1}.jpg")
            plt.savefig(output_path, dpi=300)  # Save with high resolution

            # Close the figure after saving to free memory
            plt.close()