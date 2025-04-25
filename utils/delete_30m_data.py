import pandas as pd


def delete_30min_records(csv_file_path):
    """
    Delete rows where record_time is at 30-minute marks and save back to the original file

    Args:
        csv_file_path (str): Path to the CSV file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)

        # Convert record_time to datetime if it's not already
        df["record_time"] = pd.to_datetime(df["record_time"])

        # Keep only rows where minutes are not 30
        df_filtered = df[df["record_time"].dt.minute != 30]

        # Save the filtered data back to the original file
        df_filtered.to_csv(csv_file_path, index=False)
        print(f"Successfully processed {csv_file_path}")
        print(f"Removed {len(df) - len(df_filtered)} rows with 30-minute intervals")

    except Exception as e:
        print(f"Error processing file: {str(e)}")


if __name__ == "__main__":
    # Example usage
    # Replace with your actual CSV file path
    file_path = "..\\data\\2024_shisanling_pm2_5_raw_data.csv"
    delete_30min_records(file_path)
