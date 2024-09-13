# Use the official Python image as the base image
FROM python:3.12

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements.txt file and install dependencies
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the rest of the application files into the container
COPY ./app /code/app

# Expose the port that the backend application will run on
EXPOSE 8000

# Command to run the backend application
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]