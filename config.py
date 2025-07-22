import os
startDate = "20250701"
endDate = "20250714"
expiry_date = "202507018"
data_folder = os.path.join("data", expiry_date)
symbols = []
if os.path.exists(data_folder):
    for fname in os.listdir(data_folder):
        if fname.startswith("MARK:") and fname.endswith(".csv"):
            symbol = fname.replace("MARK:", "").replace(".csv", "")
            symbols.append(symbol)
else:
    print(f"[WARNING] Data folder not found: {data_folder}")
