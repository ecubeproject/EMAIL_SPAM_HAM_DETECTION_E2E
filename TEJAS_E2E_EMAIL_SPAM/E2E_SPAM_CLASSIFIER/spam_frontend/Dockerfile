# Use the official Python image as the base image
FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install streamlit package
RUN pip install streamlit

# Copy the rest of the application files into the container
COPY . /app

# Expose the port that the front-end application will run on
EXPOSE 8501

# Entry point for running the Streamlit application
ENTRYPOINT ["streamlit", "run"]

# Default command to start the Streamlit application
CMD ["app.py"]
