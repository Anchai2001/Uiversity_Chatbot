import pandas as pd
from datetime import datetime, time

def load_schedule_data(file_path):
    """
    Load and preprocess university schedule data from an Excel file.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: Processed schedule data with standardized formats.

    Raises:
        FileNotFoundError: If the Excel file does not exist.
        KeyError: If required columns are missing in the Excel file.
        ValueError: If critical data preprocessing steps fail.
    """
    # Validate file existence
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
    except FileNotFoundError:
        raise FileNotFoundError(f"Excel file not found at: {file_path}")
    except Exception as e:
        raise ValueError(f"Failed to load Excel file: {str(e)}")

    # Validate required columns
    required_columns = ['Course', 'Teacher', 'Day', 'Time', 'Room', 'Exam Type', 'Exam Date', 'Exam Time']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in Excel file: {', '.join(missing_columns)}")

    # Clean and preprocess data
    df.fillna('', inplace=True)

    # Convert exam dates to datetime objects
    df['Exam Date'] = pd.to_datetime(
        df['Exam Date'],
        format='%d.%m.%Y',
        errors='coerce'
    )

    # Standardize exam types
    exam_type_mapping = {
        'Lecture': 'Lecture',
        'Midterm': 'Midterm',
        'Final': 'Final',
        'Makeup': 'Makeup'
    }
    df['Exam Type'] = df['Exam Type'].map(exam_type_mapping).fillna(df['Exam Type'])

    # Clean time strings: replace '.' with ':' and handle ranges
    def clean_time(x):
        if isinstance(x, str) and x.strip():
            # Replace dots with colons
            x = x.replace('.', ':')
            # If it's a range (e.g., "08:00-10:00"), take the start time
            if '-' in x:
                x = x.split('-')[0].strip()
            # Debug: Log cleaned time
            print(f"Cleaned time: '{x}'")
            return x
        return ''

    # Apply time cleaning to both Time and Exam Time
    df['Time'] = df['Time'].apply(clean_time)
    df['Exam Time'] = df['Exam Time'].apply(clean_time)

    # Convert times to datetime.time objects
    def to_time_obj(time_str):
        if time_str and isinstance(time_str, str):
            try:
                # Try HH:MM:SS format first
                dt = pd.to_datetime(time_str, format='%H:%M:%S', errors='coerce')
                if not pd.isna(dt):
                    result = dt.time()
                    print(f"Parsed '{time_str}' as {result} (HH:MM:SS)")
                    return result
                # Fallback to HH:MM format
                dt = pd.to_datetime(time_str, format='%H:%M', errors='coerce')
                if not pd.isna(dt):
                    result = dt.time()
                    print(f"Parsed '{time_str}' as {result} (HH:MM)")
                    return result
                print(f"Failed to parse time: '{time_str}'")
                return None
            except Exception as e:
                print(f"Error parsing '{time_str}': {str(e)}")
                return None
        return None

    for col in ['Time', 'Exam Time']:
        df[col] = df[col].apply(to_time_obj)

    return df

if __name__ == "__main__":
    try:
        df = load_schedule_data('cleaned_schedule.xlsx')
        print("Data loaded successfully!")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)
    except Exception as e:
        print(f"Error: {str(e)}")