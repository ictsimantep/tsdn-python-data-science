# Use the official Python image
FROM python:3.9-slim

# Set a working directory inside the container
WORKDIR /app

# Copy all files to the working directory
COPY . /app

# Install required Python libraries
RUN pip install flask joblib xgboost numpy pandas gunicorn

# Expose the port for the Flask app
EXPOSE 5000

# Use Gunicorn to serve the app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
