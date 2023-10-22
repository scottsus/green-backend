# Use the official Python image as a parent image
FROM python:3.9-slim

# Set the working directory within the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Define environment variables (optional)
ENV FLASK_APP=server.py
# ENV FLASK_RUN_HOST=0.0.0.0
ENV OPENAI_API_KEY=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

# Expose port 5000 for the Flask app to listen on
EXPOSE 5000

# Command to run the Flask application
CMD ["flask", "run"]
