from flask import Flask, render_template, request, redirect, url_for
from src.pipelines.prediction_pipeline import CustomData, Predict
import subprocess


application = Flask(__name__)
app = application


numerical_cols = ["delivery_person_Age", "delivery_person_ratings",
                  "restaurant_latitude", "restaurant_longitude", "delivery_location_latitude",
                  'delivery_location_longitude', "multiple_deliveries", 'vehicle_condition',
                  "time_orderd (Ex: 17:00)", "time_order_picked (Ex: 17:15)"]
categorical_cols = ["city", "type_of_vehicle",
                    "type_of_order", "festival", "weather_conditions", "road_traffic_density"]
columns = categorical_cols + numerical_cols

city_columns = ["Urban", "Semi-Urban", "Metropolitian"]
type_of_vehicle_columns = ["bicycle", "electric_scooter", "scooter", "motorcycle"]
type_of_order_columns = ["Drinks", "Snack", "Meal", "Buffet"]
festival_columns = ["No", "Yes"]
road_traffic_density_columns = ["Low", "Medium", "High", "Jam"]
weather_conditions_columns = ['Cloudy', 'Sunny', 'Windy', 'Fog', 'Sandstorms', 'Stormy']
drop_down_order = [city_columns, type_of_vehicle_columns, type_of_order_columns, festival_columns,
                   weather_conditions_columns, road_traffic_density_columns]


@app.route("/")
def index():
    return render_template("index.html", col=columns, drop_down_order=drop_down_order,
                           categorical_cols=categorical_cols, enumerate=enumerate)


@app.route('/train', methods=['GET'])
def train_model():
    # Call the training pipeline script using subprocess
    subprocess.run(['python', 'src/pipelines/training_pipeline.py'])
    return redirect(url_for("index"))


@app.route("/predict", methods=["GET", "POST"])
def new_prediction():
    delivery_person_Age = int(request.form["delivery_person_Age"])
    delivery_person_Ratings = float(request.form["delivery_person_ratings"])
    restaurant_latitude = float(request.form["restaurant_latitude"])
    restaurant_longitude = float(request.form["restaurant_longitude"])
    delivery_location_latitude = float(request.form["delivery_location_latitude"])
    delivery_location_longitude = float(request.form["delivery_location_longitude"])
    time_orderd = request.form["time_orderd (Ex: 17:00)"]
    time_order_picked = request.form["time_order_picked (Ex: 17:15)"]
    vehicle_condition = int(request.form["vehicle_condition"])
    multiple_deliveries = int(request.form["multiple_deliveries"])
    weather_conditions = request.form["weather_conditions"]
    road_traffic_density = request.form["road_traffic_density"]
    type_of_vehicle = request.form["type_of_vehicle"]
    type_of_order = request.form["type_of_order"]
    festival = request.form["festival"]
    city = request.form["city"]
    data = CustomData(delivery_person_age=delivery_person_Age, delivery_person_ratings=delivery_person_Ratings,
                      restaurant_latitude=restaurant_latitude, restaurant_longitude=restaurant_longitude,
                      delivery_location_latitude=delivery_location_latitude,
                      delivery_location_longitude=delivery_location_longitude,
                      time_orderd=time_orderd, time_order_picked=time_order_picked,
                      weather_conditions=weather_conditions, road_traffic_density=road_traffic_density,
                      vehicle_condition=vehicle_condition, type_of_vehicle=type_of_vehicle,
                      type_of_order=type_of_order, multiple_deliveries=multiple_deliveries, festival=festival,
                      city=city)
    df = data.get_data_as_dataframe()
    model = Predict()
    prediction = model.predict(df)
    return render_template("result.html", predict=int(round(prediction[0], 0)))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
