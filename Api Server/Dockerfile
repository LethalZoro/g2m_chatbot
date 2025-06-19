# Use an official Python 3.13.2 runtime as a parent image
FROM python:3.13.2-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container at /app
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Run uvicorn when the container launches
# The host 0.0.0.0 is important to make the container accessible from the outside
CMD uvicorn main:app --host 0.0.0.0 --port $PORT