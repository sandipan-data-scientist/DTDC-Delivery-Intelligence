# use a slim Python image to keep the container small
FROM python:3.11-slim

# set the working directory inside the container
WORKDIR /app

# copy requirements first so Docker can cache the pip install layer
# if requirements.txt has not changed, Docker will not reinstall packages
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy the application code and saved models
COPY main.py .
COPY models/ ./models/

# expose the port the API will run on
EXPOSE 8000

# start the API when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]