# Use an official Python runtime as a parent image
FROM python:3.10-slim
# Set the working directory in the container
WORKDIR /usr/src/app
# Copy the current directory contents into the container at /usr/src/app
COPY . .
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Install additional system dependencies required for your project
RUN apt-get update && \
   apt-get install -y \
   libgl1-mesa-glx \
   libglib2.0-0 \
&& rm -rf /var/lib/apt/lists/*
# Expose port if your app runs on a specific port
# EXPOSE 8000
# Define environment variable
ENV PYTHONUNBUFFERED=1
# Command to run your script
CMD ["python", "main_script.py", "your_arxiv_link_here"]