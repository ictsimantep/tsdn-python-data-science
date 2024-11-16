import psycopg2
from datetime import datetime

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="tsdn", user="postgres", password="cARQcHPP_K", host="103.196.153.247", port="5432"
)
cursor = conn.cursor()

# Data to insert
data = [
    ("2024-11-06 11:34:01", "Sahalah", "L", 26, 76, 177, 137, 84, 74, "Content Creator"),
    ("2024-11-06 11:36:09", "Yudi", "L", 21, 46, 164, 108, 81, 81, "Content Creator"),
    ("2024-11-06 11:38:09", "Akbar", "L", 17, 60, 169, 128, 62, 54, "Pelajar"),
    ("2024-11-06 11:42:12", "M.Ibnu.Fajar", "L", 25, 63, 169, 121, 73, 86, "Advertiser"),
]

# Insert data with calculated BMI and risk
for row in data:
    timestamp, nama, jenis_kelamin, umur, bb, tb, systol, diastol, heart_rate, profesi = row
    bmi = bb / ((tb / 100.0) ** 2)
    risk = "High Risk" if systol > 130 or diastol > 80 else "Low Risk"
    recommendation_food = "Low sodium diet" if risk == "High Risk" else "Balanced diet"
    recommendation_sport = "Jogging" if risk == "High Risk" else "Walking"
    recommendation_medicine = "Antihypertensive" if risk == "High Risk" else "None"

    cursor.execute(
        """
        INSERT INTO health_data (
            timestamp, nama, jenis_kelamin, umur, bb, tb, systol, diastol, heart_rate, profesi, risk, bmi, 
            recommendation_food, recommendation_sport, recommendation_medicine
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            timestamp, nama, jenis_kelamin, umur, bb, tb, systol, diastol, heart_rate, profesi, risk, bmi,
            recommendation_food, recommendation_sport, recommendation_medicine
        ),
    )

# Commit and close
conn.commit()
cursor.close()
conn.close()
