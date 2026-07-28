import os
import zipfile
import pandas as pd
import numpy as np
print("-"*70)
print("CHECKING STREAMFLOW DATA QUALITY FOR ALL CAMELS BASINS")
print("-"*70)
# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
import requests
url = "https://github.com/saeedbalahang-collab/Flood_Project/releases/download/v1.0.0/CAMELS_Streamflow.zip"
response = requests.get(url)
with open("CAMELS_Streamflow.zip", "wb") as f:
    f.write(response.content)
print("Downloaded!")
import zipfile
with zipfile.ZipFile("CAMELS_Streamflow.zip", 'r') as zip_ref:
    zip_ref.extractall("CAMELS_Streamflow")

print("ZIP extracted successfully")

EXTRACT_DIR = "/content/CAMELS_Streamflow"
# ------------------------------------------------------------
# Find txt files
# ------------------------------------------------------------

txt_files = []

for root, dirs, files in os.walk(EXTRACT_DIR):

    for f in files:

        if f.lower().endswith(".txt"):

            txt_files.append(
                os.path.join(root, f)
            )


print(f"Total txt files found: {len(txt_files)}")


# ------------------------------------------------------------
# Complete CAMELS-US daily period
# ------------------------------------------------------------

FULL_DATES = pd.date_range(
    start="1980-10-01",
    end="2014-09-30",
    freq="D"
)


EXPECTED_RECORDS = len(FULL_DATES)

print(
    f"Expected records per basin: {EXPECTED_RECORDS}"
)


# ------------------------------------------------------------
# Quality assessment
# ------------------------------------------------------------

summary = []


for file in txt_files:

    try:

        gauge_id = os.path.basename(file).split(".")[0]


        # -----------------------------
        # Read streamflow file
        # -----------------------------

        df = pd.read_csv(
            file,
            sep=r"\s+",
            header=None,
            names=[
                "gauge_id",
                "year",
                "month",
                "day",
                "Q",
                "Flag"
            ]
        )


        # -----------------------------
        # Create date column
        # -----------------------------

        df["Date"] = pd.to_datetime(
            df[["year","month","day"]]
        )


        # Keep only required columns

        df = df[
            [
                "Date",
                "Q"
            ]
        ]


        # -----------------------------
        # Create complete time series
        # -----------------------------

        complete_df = pd.DataFrame(
            {
                "Date": FULL_DATES
            }
        )


        complete_df = complete_df.merge(
            df,
            on="Date",
            how="left"
        )


        # -----------------------------
        # Identify missing values
        # -----------------------------

        missing_mask = (
            complete_df["Q"].isna()
            |
            (complete_df["Q"] == -999)
        )


        missing_records = (
            missing_mask.sum()
        )


        valid_records = (
            EXPECTED_RECORDS -
            missing_records
        )


        missing_percent = (
            100 *
            missing_records /
            EXPECTED_RECORDS
        )



        ################
        # ------------------------------------------------------------
        # 2012-2014 Quality Assessment
        # ------------------------------------------------------------

        MODEL_DATES = pd.date_range(
            start="2012-01-01",
            end="2014-12-31",
            freq="D"
        )

        MODEL_EXPECTED_RECORDS = len(MODEL_DATES)


        model_df = pd.DataFrame(
            {
                "Date": MODEL_DATES
            }
        )


        model_df = model_df.merge(
            df,
            on="Date",
            how="left"
        )


        model_missing_mask = (
            model_df["Q"].isna()
            |
            (model_df["Q"] == -999)
        )


        model_missing_records = (
            model_missing_mask.sum()
        )


        model_valid_records = (
            MODEL_EXPECTED_RECORDS -
            model_missing_records
        )


        model_missing_percent = (
            100 *
            model_missing_records /
            MODEL_EXPECTED_RECORDS
        )



        # -----------------------------
        # Save results
        # -----------------------------
        summary.append({

          "gauge_id": gauge_id,

          "Expected_Records":
              EXPECTED_RECORDS,

          "Available_File_Records":
              len(df),

          "Valid_Records":
              valid_records,

          "Missing_Records":
              missing_records,

          "Missing_Percent":
              round(
                  missing_percent,
                  3
              ),

          "2012_2014_Expected_Records":
              MODEL_EXPECTED_RECORDS,

          "2012_2014_Valid_Records":
              model_valid_records,

          "2012_2014_Missing_Records":
              model_missing_records,

          "2012_2014_Missing_Percent":
              round(
                  model_missing_percent,
                  3
              )

      })



    except Exception as e:

        print(
            f"Error processing {file}: {e}"
        )



# ------------------------------------------------------------
# Create dataframe
# ------------------------------------------------------------

quality_df = pd.DataFrame(summary)


quality_df = quality_df.sort_values(
    "Missing_Percent",
    ascending=False
)


# ------------------------------------------------------------
# Threshold summaries
# ------------------------------------------------------------

under10 = quality_df[
    quality_df["Missing_Percent"] < 10
]


under20 = quality_df[
    quality_df["Missing_Percent"] < 20
]


print("-"*70)
print("SUMMARY")
print("-"*70)


print(
    f"Total gauges: {len(quality_df)}"
)


print(
    f"<10% missing: {len(under10)}"
)


print(
    f"<20% missing: {len(under20)}"
)


print(
    f">=20% missing: "
    f"{(quality_df['Missing_Percent']>=20).sum()}"
)


display(
    quality_df.head(20)
)



# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

OUTPUT_DIR = (
    "/content/Results/Streamflow_Quality"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


quality_df.to_csv(
    f"{OUTPUT_DIR}/All_Gauge_Streamflow_Quality.csv",
    index=False
)


under10.to_csv(
    f"{OUTPUT_DIR}/Gauge_LessThan10PercentMissing.csv",
    index=False
)


under20.to_csv(
    f"{OUTPUT_DIR}/Gauge_LessThan20PercentMissing.csv",
    index=False
)
print("-"*70)
print("FILES SAVED")
print("-"*70)

print(OUTPUT_DIR)
