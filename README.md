DTDC Delivery Intelligence API

What this is

A production-ready REST API that exposes machine learning models trained on DTDC shipment data.
It predicts delivery delays, transit durations, provides demand-based pricing advice,
and city-level demand forecasts.

Project Structure

dtdc-api/
    main.py                 the FastAPI application
    Dockerfile              container definition
    requirements.txt        Python dependencies
    models/
        delay_classifier.pkl
        transit_regressor.pkl
        feature_scaler.pkl
        le_mode.pkl
        le_nature.pkl
        le_payment.pkl
        route_stats.csv
        city_forecasts.csv

Step 1: Train and save the models

Run the Jupyter notebook end to end. This will create the models/ folder
with all the .pkl and .csv files the API needs.

Step 2: Build the Docker image

Run this command from the folder containing the Dockerfile:

    docker build -t dtdc-api .

Step 3: Start the container

    docker run -d -p 8000:8000 --name dtdc-api dtdc-api

The API is now running at http://localhost:8000

Step 4: View the interactive API documentation

Open your browser and go to:
    http://localhost:8000/docs

FastAPI generates a Swagger UI automatically from the code.

Step 5: Call the endpoints

Predict delay for a new shipment (using curl):

    curl -X POST http://localhost:8000/predict-delay \
      -H "Content-Type: application/json" \
      -d '{
        "chargeable_wt": 5.5,
        "actual_wt": 4.8,
        "volumetric_wt": 6.2,
        "total_pieces": 1,
        "tariff": 350.0,
        "vas_charges": 45.0,
        "total_amount": 420.0,
        "mode": "Surface",
        "nature": "Non-Dox",
        "payment": "Cash",
        "is_b2b": 0,
        "has_vas": 1,
        "is_inter_state": 1,
        "day_of_week": 0,
        "week_number": 26,
        "month": 6
      }'

Get pricing advice for a route:

    curl http://localhost:8000/pricing-advice/Mumbai/Delhi

Get city demand forecast:

    curl http://localhost:8000/city-forecast/Pune

Stop the container when done:

    docker stop dtdc-api
    docker rm dtdc-api

Environment variables (optional)

You can override the port by changing the -p flag when running docker run.
For example -p 9090:8000 will expose the API on your machine's port 9090.

Updating the models

1. Retrain the models by re-running the notebook on new data.
2. The new .pkl files will overwrite the old ones in models/.
3. Rebuild the Docker image: docker build -t dtdc-api .
4. Restart the container.


The /docs page gives you a form you can fill in with your browser.
The models/ folder is what makes the API intelligent. Without it the container will fail to start.
Always run the notebook fully before building the Docker image.