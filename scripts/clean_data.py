import pandas as pd

#define input and output path
RAW = "data/raw/activities_sample.csv"; OUT = "data/processed/activities_clean.csv"

#raw data
df = pd.read_csv(RAW)
print("Before",df.shape)

df.columns = [c.strip().lower() for c in df.columns]
num_cols = ["student_id","grade level","gpa","attendance_rate","club_interest","participated"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors = "coerce")

df = df.dropna(subset = ["student_id"])
df["student_id"] = df["student_id"].astype("int64")

#define proper data
if "gpa" in df: df["gpa"] = df["gpa"].clip(0, 4)
if "attendance_rate" in df: df["attendance_rate"] = df["attendance_rate"].clip(0, 1)
if "club_interest" in df: df["club_interest"] = df["club_interest"].clip(0, 5)
if "grade_level" in df: df["grade_level"] = df["grade_level"].clip(9, 12)
if "participated" in df: df["participated"] = df["participated"].clip(0, 1)

fill_map = {}
if "gpa" in df: fill_map["gpa"] = df["gpa"].median()
if "attendance_rate" in df: fill_map["attendance_rate"] = df["attendance_rate"].median()
if "club_interest" in df: fill_map["club_interest"] = df["club_interest"].median()
df = df.fillna(value=fill_map)

astype_map = {}
if "grade_level" in df: astype_map["grade_level"] = "int64"
if "club_interest" in df: astype_map["club_interest"] = "int64"
if "participated" in df: astype_map["participated"] = "int64"
df = df.astype(astype_map)

print("After:", df.shape)
print("Missing ratio:\n", df.isna().mean().sort_values(ascending=False).head(10))

df.to_csv(OUT, index=False)
print("Saved to:", OUT)

