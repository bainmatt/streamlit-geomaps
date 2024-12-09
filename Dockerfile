# To build a Docker image:
#
#   $ docker build -t streamlit-geomaps .
#
# To run the Docker container:
#
#   $ docker run -p 8501:8501 -v $(pwd):/app streamlit-geomaps
#

# Base image
FROM python:3.11-slim

# Set working directory (/src)
WORKDIR /app

# Copy files (. src/app)
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements_app.txt

# Expose Streamlit default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "src/app.py"]