import pandas as pd

def load_incident_data(path="assets/incidents_with_causes.json"):
    # Load the JSON file into a DataFrame
    df = pd.read_json(path)
    
    # Fill NaN values with empty strings for safer concatenation
    df.fillna("", inplace=True)

    # Ensure that all necessary columns exist in the dataframe
    required_columns = ["incident_id", "description", "category", "urgency", "ci_id", "cr_number", "resolution", "tags", "incident_date", "cause"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # If 'tags' is a list, convert them to a comma-separated string
    df["tags"] = df["tags"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

    # Combine relevant columns into a new 'combined_text' column
    df["combined_text"] = (
        "Incident ID: " + df["incident_id"] + ". " +
        "Description: " + df["description"] + ". " +
        "Category: " + df["category"] + ". " +
        "Urgency: " + df["urgency"] + ". " +
        "CI ID: " + df["ci_id"] + ". " +
        "CR Number: " + df["cr_number"] + ". " +
        "Resolution: " + df["resolution"] + ". " +
        "Tags: " + df["tags"] + ". " +
        "Cause: " + df["cause"] + ". " +
        "Incident Date: " + df["incident_date"]
    )
    print(load_incident_data().head(), df["combined_text"].tolist(),)
    return df

print(load_incident_data().head())