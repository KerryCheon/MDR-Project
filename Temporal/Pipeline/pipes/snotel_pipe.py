# Jakob Balkovec & Kerry Cheon
# SNOTEL Pipe

# This module defines the SNOTELPipe class, which handles parsing and processing
# of SNOTEL .stm files into a unified DataFrame format compatible with the pipeline.

import os
import re
import pandas as pd
from pathlib import Path
from glob import glob
from collections import defaultdict
from ..utils.logger import get_logger
from ..utils.config import load_config


class SNOTELPipe:
    """
    Parses SNOTEL .stm files and converts them to a unified DataFrame.
    """

    def __init__(self, config=None):
        # pre:  config contains SNOTEL-specific parsing parameters
        # post: initializes SNOTELPipe with configuration
        # desc: Sets up SNOTEL data parsing with station metadata

        self.config = config or load_config()
        parse_cfg = self.config

        self.in_dir = Path(parse_cfg.get("in_dir", "data/raw"))
        self.out_dir = Path(parse_cfg.get("out_dir", "data/processed"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.latitude = parse_cfg.get("latitude")
        self.longitude = parse_cfg.get("longitude")
        self.elevation = parse_cfg.get("elevation")
        self.station_name = parse_cfg.get("station", "unknown_station")

        self.logger = get_logger().getChild(f"snotel.{self.station_name}")

    def parse_filename(self, filepath):
        """
        Extract variable, depth, sensor type, and channel from .stm filename.

        Example filename patterns:
        - sm_0.050800_0.050800_Hydraprobe-Analog-A_... (soil moisture with sensor A)
        - ts_0.050800_0.050800_Thermistor-A_... (temperature sensor A)

        Returns:
            tuple: (variable, depth_upper, depth_lower, sensor, suffix)
        """
        filename = os.path.basename(filepath)

        # Pattern: _variable_depth_depth_sensor-suffix_
        match = re.search(r"_(\w+)_([\d.-]+)_([\d.-]+)_([\w\-]+?)(?:-([A-D]))?_", filename)
        if match:
            variable = match.group(1)      # e.g., sm, ts, ta

            if "_" in variable:
                variable = variable.split("_")[-1]

            depth_upper = match.group(2)   # e.g., 0.050800
            depth_lower = match.group(3)   # e.g., 0.050800
            sensor = match.group(4)        # e.g., Hydraprobe-Analog, Thermistor
            suffix = match.group(5)        # e.g., A, B, C, D or None
            return variable, depth_upper, depth_lower, sensor, suffix
        else:
            return None, None, None, None, None

    def load_stm_file(self, filepath):
        """
        Load a single .stm file into a DataFrame.

        .stm file format (space-separated):
        YYYY/MM/DD HH:MM:SS Value Quality Validation

        Returns:
            DataFrame with columns: Date, Time, Value
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()

            data_rows = []
            for line in lines:
                # Match lines starting with date pattern
                if re.match(r"^\d{4}/\d{2}/\d{2}", line):
                    parts = line.split()
                    if len(parts) >= 5:
                        # Extract: Date, Time, Value (ignore Quality, Validation)
                        data_rows.append(parts[:3])

            df = pd.DataFrame(data_rows, columns=["Date", "Time", "Value"])
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            return df

        except Exception as e:
            self.logger.error(f"Failed to load {filepath}: {e}")
            return pd.DataFrame()

    def build_column_name(self, variable, depth_upper):
        """
        Create standardized column names based on variable type and depth.

        Args:
            variable: Variable code (sm, ts, ta, swe, etc.)
            depth_upper: Depth in meters (e.g., 0.050800)

        Returns:
            Standardized column name (e.g., soil_moisture_5cm, soil_temp_5cm)
        """
        depth_cm = float(depth_upper) * 100 if depth_upper else None

        # Map SNOTEL variable codes to standardized names
        variable_mapping = {
            'sm': 'soil_moisture',
            'ts': 'soil_temp',
            'ta': 'air_temp',
            'swe': 'snow_water_equiv',
            'sweq': 'snow_water_equiv',
            'precip': 'precipitation',
            'rh': 'relative_humidity'
        }

        base = variable_mapping.get(variable, variable)

        if depth_cm is not None and depth_cm > 0:
            col_name = f"{base}_{int(depth_cm)}cm"
        else:
            col_name = base

        return col_name

    def add_rolling_rain_features(self, df):
        """
        Add 3-day rolling sum for all precipitation columns.

        Args:
            df: DataFrame with DateTime column and precipitation data

        Returns:
            DataFrame with additional *_3day rolling columns
        """
        self.logger.info("Adding 3-day rolling rain features...")

        # Parse DateTime for rolling calculations
        df['DateTime_parsed'] = pd.to_datetime(df['DateTime'], errors='coerce')
        df = df.sort_values('DateTime_parsed').reset_index(drop=True)

        # Find all precipitation-related columns
        rain_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['precip', 'rain', 'ppt']):
                rain_columns.append(col)

        # Also check for satellite rain
        if 'Rain_sat' in df.columns:
            rain_columns.append('Rain_sat')

        rain_columns = list(set(rain_columns))

        if not rain_columns:
            self.logger.warning("No precipitation columns found. Skipping rolling rain features.")
            df = df.drop(columns=['DateTime_parsed'])
            return df

        self.logger.info(f"Found {len(rain_columns)} precipitation column(s): {', '.join(rain_columns)}")

        # Calculate 3-day rolling sum for each rain column
        rolling_cols_added = []
        for col in rain_columns:
            new_col = f"{col}_3day"
            df[new_col] = df[col].rolling(window=3, min_periods=1).sum()
            rolling_cols_added.append(new_col)

            original_mean = df[col].mean()
            rolling_mean = df[new_col].mean()
            self.logger.debug(f"{new_col}: Original mean={original_mean:.2f}, Rolling mean={rolling_mean:.2f}")

        df = df.drop(columns=['DateTime_parsed'])
        self.logger.info(f"Added {len(rolling_cols_added)} rolling rain column(s)")

        return df

    def run(self, _=None):
        """
        Main execution method for SNOTEL data processing.

        Process flow:
        1. Find all .stm files in input directory
        2. Group by variable and depth (averaging multiple sensors)
        3. Merge all variables into single DataFrame
        4. Add metadata (station ID, coordinates, elevation)
        5. Optionally add rolling rain features

        Returns:
            DataFrame with all processed SNOTEL data
        """
        stm_files = glob(str(self.in_dir / "*.stm"))

        if not stm_files:
            self.logger.warning(f"No .stm files found in {self.in_dir}")
            return pd.DataFrame()

        self.logger.info(f"[{self.station_name}] Found {len(stm_files)} .stm files")

        # Group files by variable + depth (ignoring A/B/C/D sensor suffixes)
        groups = defaultdict(list)
        for filepath in stm_files:
            variable, depth_upper, depth_lower, sensor, suffix = self.parse_filename(filepath)
            if variable is None:
                self.logger.warning(f"Skipping unknown file pattern: {filepath}")
                continue
            key = (variable, depth_upper)
            groups[key].append(filepath)

        master_df = None

        # Process each variable/depth group
        for (variable, depth_upper), files in groups.items():
            self.logger.info(f"Processing {variable} at depth {depth_upper}m with {len(files)} sensor(s)")

            # Load and stack all sensor files (A/B/C/D)
            dfs = []
            for filepath in files:
                df = self.load_stm_file(filepath)
                if not df.empty:
                    dfs.append(df[['Date', 'Time', 'Value']])

            if not dfs:
                self.logger.warning(f"No valid data for {variable} at {depth_upper}m")
                continue

            # Concatenate and average duplicate Date/Time entries (for multiple sensors)
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.groupby(['Date', 'Time'], as_index=False).mean()

            # Create DateTime column
            combined_df['DateTime'] = combined_df['Date'] + ' ' + combined_df['Time']
            combined_df = combined_df.drop(columns=['Date', 'Time'])

            # Create standardized column name
            col_name = self.build_column_name(variable, depth_upper)
            combined_df = combined_df.rename(columns={'Value': col_name})

            # Merge into master DataFrame
            if master_df is None:
                master_df = combined_df
            else:
                master_df = pd.merge(master_df, combined_df, on='DateTime', how='outer')

        if master_df is None or master_df.empty:
            self.logger.error("No data was successfully parsed from .stm files")
            return pd.DataFrame()

        # Sort by DateTime
        master_df = master_df.sort_values('DateTime').reset_index(drop=True)

        # Add metadata columns
        master_df['date'] = pd.to_datetime(master_df['DateTime'], errors='coerce').dt.strftime('%Y-%m-%d')
        master_df['station_id'] = self.station_name

        # edit: make it strongly typed
        if self.latitude is not None:
            master_df["latitude"] = float(self.latitude)
        if self.longitude is not None:
            master_df["longitude"] = float(self.longitude)
        if self.elevation is not None:
            master_df["elevation"] = float(self.elevation)

        # Reorder columns: DateTime, date, station_id, lat, lon, then data columns, elevation last
        priority_cols = ['DateTime', 'date', 'station_id']
        if self.latitude is not None:
            priority_cols.append('latitude')
        if self.longitude is not None:
            priority_cols.append('longitude')

        data_cols = [col for col in master_df.columns if col not in priority_cols and col != 'elevation']

        final_order = priority_cols + data_cols
        if self.elevation is not None:
            final_order.append('elevation')

        master_df = master_df[final_order]

        # Convert date to datetime format for pipeline compatibility
        master_df['date'] = pd.to_datetime(master_df['date'], errors='coerce')

        # Add rolling rain features if configured
        if self.config.get("add_rolling_rain", True):
            try:
                master_df = self.add_rolling_rain_features(master_df)
            except Exception as e:
                self.logger.warning(f"Rolling rain calculation failed: {e}")

        self.logger.info(
            f"[{self.station_name}] SNOTEL parsing complete: "
            f"{len(master_df)} rows, {len(master_df.columns)} columns"
        )

        return master_df
