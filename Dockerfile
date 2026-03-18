# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application source code
# We copy the entire project context to ensure imports work correctly
COPY . .

# Expose the port that Gunicorn will run on (Cloud Run defaults to 8080)
ENV PORT=8080
EXPOSE 8080

# Define the command to run the application using Gunicorn
# app.dashboard:server points to the 'server' object in app/dashboard.py
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app.dashboard:server
